"""Paper-motivated nanochat adaptations for pretraining, not cached decoding.

DCMHA follows Appendix D of arXiv:2405.08553 (rank 2).
FoX uses cumulative log-forget gates (arXiv:2503.02130), without RoPE.
DIFF V2 uses shared K/V, extra Q, token/head sigmoid lambda, no output norm.
FoPE implements Eq.13: fixed Fourier perturbations and zero low frequencies.
CLA2 shares adjacent-layer projected K/V, retaining recipient value embeddings.
These are controlled nanochat adaptations, not full paper reproductions.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from nanochat.gpt import GPT, GPTConfig, Linear, norm, apply_rotary_emb
from nanochat.common import COMPUTE_DTYPE
from nanochat.flash_attention import flash_attn

VARIANTS = ('baseline', 'cla2', 'fox', 'dcmha', 'diff_v2', 'fope', 'fox_rope')


class Compose(nn.Module):
    def __init__(self, width, heads, rank=2):
        super().__init__()
        self.heads, self.rank = heads, rank
        hidden = 2 * heads * rank
        self.q1 = Linear(width, hidden, bias=False)
        self.q2 = Linear(hidden, hidden, bias=False)
        self.k1 = Linear(width, hidden, bias=False)
        self.k2 = Linear(hidden, hidden, bias=False)
        self.qg = Linear(width, heads, bias=False)
        self.kg = Linear(width, heads, bias=False)
        for module in (self.qg, self.kg):
            nn.init.zeros_(module.weight)
        # Both low-rank factors must not be zero: that would kill their gradients.
        for module in (self.q2, self.k2):
            nn.init.normal_(module.weight, std=0.02)

    def factors(self, x, a, b):
        u, v = b(F.gelu(a(x))).chunk(2, dim=-1)
        shape = (*x.shape[:2], self.rank, self.heads)
        return norm(u.reshape(shape)), v.reshape(shape)

    def forward(self, scores, x):
        # Compose in fp32; do not mix -inf masked logits across heads.
        u, v = self.factors(x, self.q1, self.q2)
        h = torch.einsum('bhts,btrh->brts', scores, u.float())
        out = scores + torch.einsum('brts,btrh->bhts', h, v.float())
        u, v = self.factors(x, self.k1, self.k2)
        h = torch.einsum('bhts,bsrh->brts', scores, u.float())
        out = out + torch.einsum('brts,bsrh->bhts', h, v.float())
        out = out + scores * self.qg(x).float().tanh().transpose(1, 2).unsqueeze(-1)
        out = out + scores * self.kg(x).float().tanh().transpose(1, 2).unsqueeze(-2)
        return out


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
        if variant in ('fox', 'fox_rope'):
            self.forget = Linear(config.n_embd + 1, config.n_head, bias=False)
            nn.init.zeros_(self.forget.weight)
            with torch.no_grad():
                self.forget.weight[:, -1].fill_(3.0)
        if variant == 'diff_v2':
            self.q_other = Linear(config.n_embd, config.n_embd, bias=False)
            self.lam = Linear(config.n_embd, config.n_head, bias=False)
            nn.init.zeros_(self.lam.weight)
        if variant == 'dcmha':
            self.pre = Compose(config.n_embd, config.n_head)
            self.post = Compose(config.n_embd, config.n_head)

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
        if self.variant in ('fox', 'fox_rope', 'dcmha'):
            pos = torch.arange(t, device=x.device)
            delta = pos[:, None] - pos[None, :]
            mask = delta >= 0
            if window_size[0] >= 0:
                mask = mask & (delta <= window_size[0])
            scores = q.transpose(1, 2).float() @ k.transpose(1, 2).float().transpose(-1, -2)
            scores = scores * self.head_dim ** -0.5
            if self.variant in ('fox', 'fox_rope'):
                inputs = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
                cumulative = F.logsigmoid(self.forget(inputs).float()).cumsum(1).transpose(1, 2)
                scores = scores + cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
            else:
                scores = self.pre(scores, x)
            probs = scores.masked_fill(~mask, float('-inf')).softmax(-1)
            if self.variant == 'dcmha':
                probs = self.post(probs, x)
            y = (probs.to(v.dtype) @ v.transpose(1, 2)).transpose(1, 2)
        elif self.variant == 'diff_v2':
            q2 = self.q_other(x).view(shape)
            q2 = norm(apply_rotary_emb(q2, *cos_sin)) * 1.2
            # Adjacent paired Q heads share K/V via GQA.
            qq = torch.stack((q, q2), dim=3).flatten(2, 3)
            both = flash_attn.flash_attn_func(qq, k, v, causal=True, window_size=window_size)
            y = both[:, :, 0::2] - self.lam(x).sigmoid().unsqueeze(-1) * both[:, :, 1::2]
        else:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        return self.c_proj(y.contiguous().view(b, t, c)), shared_out


class PilotGPT(GPT):
    def install_variant(self, variant):
        assert variant in VARIANTS
        self.variant = variant
        for block in self.transformer.h:
            block.attn = PilotAttention(block.attn, self.config, variant).to(self.get_device())
        if variant == 'fope':
            # Paper Eq.13 variant: head-shared fixed basis, identity + N(0,.1).
            # Not the authors' per-head OLMo configuration; document this explicitly.
            dim = self.config.n_embd // self.config.n_head // 2
            freq = 100000 ** (-torch.arange(dim, device=self.get_device()).float() / dim)
            active = freq >= 2 * math.pi / self.config.sequence_len
            ids = active.nonzero().flatten()
            phase = torch.arange(self.rotary_seq_len, device=self.get_device()).float()[:, None] * freq[ids]
            n = ids.numel()
            eye = torch.eye(n, device=self.get_device())
            cc = eye + torch.randn(n, n, device=self.get_device()) * .1
            ss = eye + torch.randn(n, n, device=self.get_device()) * .1
            cos = torch.ones(self.rotary_seq_len, dim, device=self.get_device())
            sin = torch.zeros_like(cos)
            cos[:, ids], sin[:, ids] = phase.cos() @ cc, phase.sin() @ ss
            self.cos = cos[None, :, None].to(COMPUTE_DTYPE)
            self.sin = sin[None, :, None].to(COMPUTE_DTYPE)

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
