"""Smoke tests run before the paid sweep; failures block training."""
import torch
from nanochat.gpt import GPT
from pico.variants import VARIANTS, build


def run(device='cpu', variants=VARIANTS):
    results = []
    for variant in variants:
        model = build(variant, device=device, depth=2, width=128, seq_len=32, vocab_size=256)
        # Nonzero residual projections make causality tests non-vacuous.
        for block in model.transformer.h:
            torch.nn.init.normal_(block.attn.c_proj.weight, std=.02)
            torch.nn.init.normal_(block.mlp.c_proj.weight, std=.02)
        x = torch.randint(0, 256, (2, 24), device=device)
        altered = x.clone()
        altered[:, 12:] = torch.randint(0, 256, (2, 12), device=device)
        a, b = model(x), model(altered)
        torch.testing.assert_close(a[:, :12], b[:, :12], atol=2e-5, rtol=2e-4)
        loss = model(x, x.roll(-1, 1))
        loss.backward()
        assert torch.isfinite(loss)
        for name, p in model.named_parameters():
            assert p.grad is not None, f'{variant}: missing gradient {name}'
            assert torch.isfinite(p.grad).all(), f'{variant}: bad gradient {name}'
        if variant == 'baseline':
            upstream = GPT(model.config).to(device)
            upstream.init_weights()
            upstream.load_state_dict(model.state_dict())
            torch.testing.assert_close(a, upstream(x), atol=2e-5, rtol=2e-4)
        # Check checkpoint loading, for each selected variant.
        reloaded = build(variant, device=device, depth=2, width=128, seq_len=32, vocab_size=256)
        reloaded.load_state_dict(model.state_dict())
        torch.testing.assert_close(a, reloaded(x), atol=2e-5, rtol=2e-4)
        results.append({'variant': variant, 'loss': loss.item(), 'status': 'passed'})
    return results


if __name__ == '__main__':
    import json
    print(json.dumps(run(), indent=2))
