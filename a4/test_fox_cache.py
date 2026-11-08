"""CPU tests: cached FoX decoding must match the uncached forward pass.

Run from the repository root with nanochat importable:
    NANOCHAT_DIR=/path/to/nanochat python -m unittest a4/test_fox_cache.py -v
"""
import os
import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
NANOCHAT = os.environ.get("NANOCHAT_DIR", str(HERE.parents[2] / "nanochat"))
for path in (NANOCHAT, HERE.parent / "a3", HERE.parent / "a3" / "A3part5", HERE):
    sys.path.insert(0, str(path))

import nanochat.engine as engine
import nanochat.gpt as gpt
import fox_cache

fox_cache.install()

VOCAB, LAYERS, HEADS, WIDTH = 64, 4, 2, 64
# Short windows on three layers, full context on the last, so the window mask is exercised.
WINDOWS = [(8, 0), (8, 0), (8, 0), (-1, 0)]


def tiny_model(seed):
    torch.manual_seed(seed)
    config = gpt.GPTConfig(sequence_len=96, vocab_size=VOCAB, n_layer=LAYERS, n_head=HEADS,
                           n_kv_head=HEADS, n_embd=WIDTH, window_pattern="SSSL")
    model = gpt.GPT(config)
    model.init_weights()
    with torch.no_grad():  # nanochat zero-inits output projections, which would hide the attention
        for parameter in model.parameters():
            parameter.normal_(0, 0.3)
    model.window_sizes = list(WINDOWS)
    return model.eval()


class FakeTokenizer:
    """Special tokens sit outside the vocabulary, so they are never sampled."""
    def encode_special(self, name):
        return 1000 + sum(map(ord, name))

    def get_bos_token_id(self):
        return 999


class CachedFoXTests(unittest.TestCase):
    def test_model_is_fox(self):
        model = tiny_model(0)
        self.assertIsInstance(model, fox_cache.fox.FoXGPT)
        self.assertTrue(all(isinstance(b.attn, fox_cache.CachedFoXAttention) for b in model.transformer.h))

    def test_cached_logits_match_full_forward(self):
        model = tiny_model(0)
        ids = torch.randint(0, VOCAB, (1, 40))
        with torch.no_grad():
            full = model(ids)
            for prompt_len in (1, 7, 25):
                cache = fox_cache.FoXKVCache(batch_size=1, num_heads=HEADS, seq_len=40, head_dim=WIDTH // HEADS,
                                             num_layers=LAYERS, device="cpu", dtype=torch.float32)
                pieces = [model(ids[:, :prompt_len], kv_cache=cache)]
                pieces += [model(ids[:, i:i + 1], kv_cache=cache) for i in range(prompt_len, 40)]
                torch.testing.assert_close(torch.cat(pieces, 1), full, atol=1e-4, rtol=1e-4)

    def test_engine_greedy_matches_uncached_loop(self):
        model = tiny_model(1)
        prompt = [3, 14, 15, 9, 26, 5, 35]
        steps = 24
        rows, _ = engine.Engine(model, FakeTokenizer()).generate_batch(
            prompt, num_samples=3, max_tokens=steps, temperature=0)
        expected = list(prompt)
        with torch.no_grad():
            for _ in range(steps):
                expected.append(model(torch.tensor([expected]))[0, -1].argmax().item())
        for row in rows:
            self.assertEqual(row, expected)


if __name__ == "__main__":
    unittest.main()
