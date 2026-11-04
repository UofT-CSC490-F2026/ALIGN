"""Check that the new control differs from FoX only by rotary application."""
import torch
from pico.variants import build


def run(device):
    kwargs = dict(seed=42, device=device, depth=2, width=128, seq_len=32, vocab_size=256)
    fox = build('fox', **kwargs)
    rope = build('fox_rope', **kwargs)
    for key, value in fox.state_dict().items():
        torch.testing.assert_close(value, rope.state_dict()[key], atol=0, rtol=0)
    # Nonzero output projections expose the attention computation at initialization.
    for block in fox.transformer.h:
        torch.nn.init.normal_(block.attn.c_proj.weight, std=.02)
    rope.load_state_dict(fox.state_dict())
    x = torch.randint(0, 256, (2, 24), device=device)
    expected = fox(x)
    with_rope = rope(x)
    assert not torch.allclose(expected, with_rope, atol=1e-6, rtol=1e-6)
    # Identity rotations must reduce the new control exactly to existing FoX.
    rope.cos.fill_(1)
    rope.sin.zero_()
    torch.testing.assert_close(expected, rope(x), atol=2e-5, rtol=2e-4)
    return {'device': device, 'status': 'passed',
            'checks': ['identical initial parameters', 'RoPE changes output',
                       'identity RoPE recovers original FoX']}
