"""KV-cached decoding for A3's FoX attention (cumulative forget gates, no RoPE).

A3's FoXAttention refuses a KV cache, so nanochat's Engine (used by the chat
evaluation and by RL rollouts) cannot generate with the FoX model. This adds
the cache. FoX scores are q.k plus (C_i - C_j), where C is the running sum of
the per-head log forget gates, so the cache keeps C next to the keys and values.

Needs nanochat and a3/A3part5 (fox.py) importable. Call install() before the
model is built, then use nanochat's Engine as usual.
"""
import torch
import torch.nn.functional as F

import nanochat.engine as engine
from nanochat.gpt import norm

import fox


class FoXKVCache(engine.KVCache):
    """KVCache that also stores the running log forget-gate sum per layer."""

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        super().__init__(batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype)
        self.gate_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, device=device, dtype=torch.float32)

    def prefill(self, other):
        super().prefill(other)
        pos = other.get_pos()
        self.gate_cache[:, :, :pos] = other.gate_cache[:, :, :pos]


class CachedFoXAttention(fox.FoXAttention):
    def __init__(self, source, config, variant):
        super().__init__(source, config, variant)
        assert config.n_head == config.n_kv_head, "cached FoX decoding assumes one key/value head per query head"

    def forward(self, x, ve, cos_sin, window_size, kv_cache=None):
        if kv_cache is None:
            return super().forward(x, ve, cos_sin, window_size)
        b, t, c = x.shape
        shape = (b, t, self.n_head, self.head_dim)
        q = self.c_q(x).view(shape)
        k, v = self.c_k(x).view(shape), self.c_v(x).view(shape)
        if ve is not None:
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve.view(shape)
        q, k = norm(q) * 1.2, norm(k) * 1.2

        inputs = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        log_forget = F.logsigmoid(self.forget(inputs).float())  # (B, T, H)
        pos = kv_cache.get_pos()
        end = pos + t
        k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
        gate_cache = kv_cache.gate_cache[self.layer_idx]
        before = gate_cache[:, pos - 1] if pos else torch.zeros_like(log_forget[:, 0])
        cumulative = before.unsqueeze(1) + log_forget.cumsum(1)  # (B, T, H)
        k_cache[:, pos:end], v_cache[:, pos:end], gate_cache[:, pos:end] = k, v, cumulative

        keys, values = k_cache[:, :end], v_cache[:, :end]
        scores = q.transpose(1, 2).float() @ keys.transpose(1, 2).float().transpose(-1, -2)
        scores = scores * self.head_dim ** -0.5
        scores = scores + cumulative.transpose(1, 2).unsqueeze(-1) - gate_cache[:, :end].transpose(1, 2).unsqueeze(-2)
        delta = (pos + torch.arange(t, device=x.device))[:, None] - torch.arange(end, device=x.device)[None, :]
        mask = delta >= 0
        if window_size[0] >= 0:
            mask = mask & (delta <= window_size[0])
        probs = scores.masked_fill(~mask, float('-inf')).softmax(-1)
        y = (probs.to(v.dtype) @ values.transpose(1, 2)).transpose(1, 2)
        if self.layer_idx == kv_cache.n_layers - 1:
            kv_cache.advance(t)
        return self.c_proj(y.contiguous().view(b, t, c))


def install():
    """Patch A3's FoX model and nanochat's Engine. Call before building the model."""
    fox.FoXAttention = CachedFoXAttention
    engine.KVCache = FoXKVCache
    fox.install()
