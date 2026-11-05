"""Reuse Part 2 FoX attention with native GPT, optimizer and checkpoint formats."""
import torch
from torch.utils.checkpoint import checkpoint
import nanochat.gpt as gpt
from pico.variants import PilotAttention

NativeGPT = gpt.GPT


class FoXAttention(PilotAttention):
    """Adapt Part 2's return signature to the native GPT block."""
    def forward(self, x, ve, cos_sin, window_size, kv_cache=None):
        if kv_cache is not None:
            raise NotImplementedError("FoX evaluation uses uncached base-model completion")
        return super().forward(x, ve, cos_sin, window_size)[0]


class FoXGPT(NativeGPT):
    def __init__(self, config):
        super().__init__(config)
        for block in self.transformer.h:
            block.attn = FoXAttention(block.attn, config, "fox")

    def init_weights(self):
        super().init_weights()
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.attn.forget.weight)
            with torch.no_grad():
                block.attn.forget.weight[:, -1].fill_(3.0)


NativeBlock = gpt.Block


class CheckpointBlock(NativeBlock):
    def forward(self, *args):
        if self.training and torch.is_grad_enabled():
            return checkpoint(super().forward, *args, use_reentrant=False)
        return super().forward(*args)


def install():
    """Change only model construction in the upstream training/checkpoint modules."""
    import nanochat.checkpoint_manager as checkpoints
    gpt.Block = CheckpointBlock
    gpt.GPT = FoXGPT
    checkpoints.GPT = FoXGPT


def verify():
    """Forward/gradient and native checkpoint compatibility against Part 2."""
    from pico.variants import build
    torch.manual_seed(42)
    reference = build("fox", device="cpu", depth=2, width=128, vocab_size=256, seq_len=32)
    model = FoXGPT(reference.config).to("cpu")
    model.init_weights()
    model.load_state_dict(reference.state_dict(), strict=True)
    x = torch.arange(32).reshape(2, 16)
    y = (x + 1) % 256
    a, b = reference(x, y), model(x, y)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    a.backward(); b.backward()
    for (name, p), (other, q) in zip(reference.named_parameters(), model.named_parameters()):
        assert name == other
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)
    return {"part2_forward_and_gradients_match": True}
