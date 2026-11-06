"""Paper-motivated nanochat adaptations for pretraining, not cached decoding.

FoX uses cumulative log-forget gates (arXiv:2503.02130), without RoPE.
CLA2 shares adjacent-layer projected K/V, retaining recipient value embeddings.
These are controlled nanochat adaptations, not full paper reproductions.
"""
import torch
from torch import nn
from torch.nn import functional as F
from nanochat.gpt import GPT, GPTConfig, Linear, norm, apply_rotary_emb
from nanochat.common import COMPUTE_DTYPE
from nanochat.flash_attention import flash_attn

VARIANTS = ('baseline', 'cla2', 'fox')


class PilotAttention(nn.Module):
    def __init__(self, source, config, variant):
        super().__init__()
        self.variant = variant
        self.layer_idx = source.layer_idx
        self.n_head = source.n_head
        self.head_dim = source.head_dim
        self.c_q, self.c_proj = source.c_q, source.c_proj
        self.ve_gate = source.ve_gate
        self.ve_gate_channels = source.ve_gate_channels
        if variant != 'cla2' or self.layer_idx % 2 == 0:
            self.c_k, self.c_v = source.c_k, source.c_v
        if variant == 'fox':
            self.forget = Linear(config.n_embd + 1, config.n_head, bias=False)
            nn.init.zeros_(self.forget.weight)
            with torch.no_grad():
                self.forget.weight[:, -1].fill_(3.0)

    def forward(self, x, ve, cos_sin, window_size, shared=None):
        b, t, c = x.shape
        shape = (b, t, self.n_head, self.head_dim)
        q = self.c_q(x).view(shape)
        if self.variant == 'cla2' and self.layer_idx % 2:
            k, raw_v = shared
        else:
            k, raw_v = self.c_k(x).view(shape), self.c_v(x).view(shape)
        # Share projections BEFORE position transform/VE. Preserve each layer's VE.
        shared_out = (k, raw_v)
        v = raw_v
        if ve is not None:
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve.view(shape)
        if self.variant != 'fox':
            q = apply_rotary_emb(q, *cos_sin)
            k = apply_rotary_emb(k, *cos_sin)
        q, k = norm(q) * 1.2, norm(k) * 1.2
        if self.variant == 'fox':
            pos = torch.arange(t, device=x.device)
            delta = pos[:, None] - pos[None, :]
            mask = delta >= 0
            if window_size[0] >= 0:
                mask = mask & (delta <= window_size[0])
            scores = q.transpose(1, 2).float() @ k.transpose(1, 2).float().transpose(-1, -2)
            scores = scores * self.head_dim ** -0.5
            inputs = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
            cumulative = F.logsigmoid(self.forget(inputs).float()).cumsum(1).transpose(1, 2)
            scores = scores + cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
            probs = scores.masked_fill(~mask, float('-inf')).softmax(-1)
            y = (probs.to(v.dtype) @ v.transpose(1, 2)).transpose(1, 2)
        else:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        return self.c_proj(y.contiguous().view(b, t, c)), shared_out


class PilotGPT(GPT):
    def install_variant(self, variant):
        assert variant in VARIANTS
        self.variant = variant
        for block in self.transformer.h:
            block.attn = PilotAttention(block.attn, self.config, variant).to(self.get_device())

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        if kv_cache is not None:
            raise NotImplementedError('Pilot only: cached decoding has not been implemented.')
        _, t = idx.shape
        cs = self.cos[:, :t], self.sin[:, :t]
        x = norm(self.transformer.wte(idx).to(COMPUTE_DTYPE))
        gate = self.smear_lambda.to(x.dtype) * self.smear_gate(x[:, 1:, :24]).sigmoid()
        x = torch.cat((x[:, :1], x[:, 1:] + gate * x[:, :-1]), dim=1)
        x0, backout, shared = x, None, None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            y, shared = block.attn(norm(x), ve, cs, self.window_sizes[i], shared)
            x = x + y
            x = x + block.mlp(norm(x))
            if i == self.config.n_layer // 2:
                backout = x
        x = norm(x - self.backout_lambda.to(x.dtype) * backout)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)
        if targets is None:
            return logits
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=-1, reduction=loss_reduction)


def build(variant, seed=42, device='cuda', depth=6, width=384, seq_len=512, vocab_size=32768):
    torch.manual_seed(seed)
    config = GPTConfig(n_layer=depth, n_embd=width, n_head=width // 64,
                       n_kv_head=width // 64, sequence_len=seq_len, vocab_size=vocab_size)
    model = PilotGPT(config).to(device)
    model.init_weights()
    model.install_variant(variant)
    return model
