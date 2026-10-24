# -*- coding: utf-8 -*-
"""Tests for the vendored OPT model (G1.3, G1.6).

Tests:
- Model construction and shape validation
- Forward pass produces correct output shapes
- KV cache mechanics
- Chunked prefill equivalence
- Generation output
"""

import os
import sys
import pytest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgespec.config import OPT_CONFIGS
from edgespec.model.opt import OPTModel, KVCache


# Use a tiny config for fast testing
TINY_CONFIG = {
    "n_layers": 2,
    "n_heads": 4,
    "d_model": 64,
    "d_ffn": 256,
    "vocab_size": 1000,
    "max_positions": 128,
    "position_offset": 2,
    "activation": "relu",
    "layer_norm_eps": 1e-5,
    "bos_token_id": 2,
    "eos_token_id": 2,
    "pad_token_id": 1,
    "tie_word_embeddings": True,
    "pre_layernorm": True,
}


class TestKVCache(object):
    """Test the KV cache."""

    def test_append(self):
        cache = KVCache(4, 16, 128, device='cpu', dtype=torch.float16)
        assert cache.length == 0

        k = torch.randn(4, 5, 16, dtype=torch.float16)
        v = torch.randn(4, 5, 16, dtype=torch.float16)
        k_full, v_full = cache.append(k, v)

        assert cache.length == 5
        assert k_full.shape == (4, 5, 16)
        assert v_full.shape == (4, 5, 16)

    def test_append_multiple(self):
        cache = KVCache(4, 16, 128, device='cpu', dtype=torch.float16)

        k1 = torch.randn(4, 3, 16, dtype=torch.float16)
        v1 = torch.randn(4, 3, 16, dtype=torch.float16)
        cache.append(k1, v1)

        k2 = torch.randn(4, 2, 16, dtype=torch.float16)
        v2 = torch.randn(4, 2, 16, dtype=torch.float16)
        k_full, v_full = cache.append(k2, v2)

        assert cache.length == 5
        assert k_full.shape == (4, 5, 16)

    def test_overflow_raises(self):
        cache = KVCache(4, 16, 5, device='cpu', dtype=torch.float16)
        k = torch.randn(4, 3, 16, dtype=torch.float16)
        v = torch.randn(4, 3, 16, dtype=torch.float16)
        cache.append(k, v)

        k2 = torch.randn(4, 3, 16, dtype=torch.float16)
        v2 = torch.randn(4, 3, 16, dtype=torch.float16)
        with pytest.raises(AssertionError):
            cache.append(k2, v2)

    def test_reset(self):
        cache = KVCache(4, 16, 128, device='cpu', dtype=torch.float16)
        k = torch.randn(4, 5, 16, dtype=torch.float16)
        v = torch.randn(4, 5, 16, dtype=torch.float16)
        cache.append(k, v)
        assert cache.length == 5

        cache.reset()
        assert cache.length == 0

    def test_memory_bytes(self):
        cache = KVCache(4, 16, 128, device='cpu', dtype=torch.float16)
        expected = 2 * 4 * 128 * 16 * 2  # 2 (K+V) * heads * seq * d_head * 2 bytes
        assert cache.memory_bytes() == expected


class TestOPTModel(object):
    """Test vendored OPT model."""

    def _make_model(self, config=None, device='cpu'):
        config = config or TINY_CONFIG
        model = OPTModel(config, dtype=torch.float16)
        model = model.to(device).to(torch.float16)
        model.lm_head_weight = model.embed_tokens.weight
        model.eval()
        model.init_kv_caches(device=device)
        return model

    def test_construct(self):
        model = self._make_model()
        assert model.n_layers == 2
        assert model.n_heads == 4
        assert model.d_model == 64

    def test_forward_block_single_token(self):
        model = self._make_model()
        token = torch.tensor([[42]], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model.forward_block(token)
        assert logits.shape == (1, 1, TINY_CONFIG['vocab_size'])

    def test_forward_block_multi_token(self):
        """Test k > 1 tokens (needed for spec-decode verification)."""
        model = self._make_model()
        tokens = torch.tensor([[42, 100, 200, 300]], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model.forward_block(tokens)
        assert logits.shape == (1, 4, TINY_CONFIG['vocab_size'])

    def test_forward_block_with_attn_probs(self):
        model = self._make_model()
        tokens = torch.tensor([[42, 100]], dtype=torch.long)
        with torch.no_grad():
            logits, attn_probs = model.forward_block(tokens, return_attn_probs=True)

        assert logits.shape == (1, 2, TINY_CONFIG['vocab_size'])
        assert attn_probs is not None
        assert len(attn_probs) == TINY_CONFIG['n_layers']
        # Each attn_probs: [n_heads, n_new, seq_len]
        assert attn_probs[0].shape[0] == TINY_CONFIG['n_heads']
        assert attn_probs[0].shape[1] == 2  # n_new

    def test_chunked_prefill(self):
        model = self._make_model()
        prompt = torch.randint(10, 500, (1, 32), dtype=torch.long)
        with torch.no_grad():
            logits = model.prefill_chunked(prompt, chunk_size=8)
        assert logits.shape == (1, 32, TINY_CONFIG['vocab_size'])

    def test_prefill_then_decode(self):
        """Prefill + decode should work without errors."""
        model = self._make_model()
        prompt = torch.randint(10, 500, (1, 16), dtype=torch.long)

        with torch.no_grad():
            logits = model.prefill_chunked(prompt, chunk_size=8)
            next_token = logits[0, -1, :].argmax(dim=-1).item()

            for _ in range(5):
                input_ids = torch.tensor([[next_token]], dtype=torch.long)
                step_logits, _ = model.forward_block(input_ids)
                next_token = step_logits[0, -1, :].argmax(dim=-1).item()

        assert model.kv_caches[0].length == 16 + 5

    def test_generate_greedy(self):
        model = self._make_model()
        prompt = torch.randint(10, 500, (1, 8), dtype=torch.long)
        with torch.no_grad():
            ids = model.generate_greedy(prompt, max_new_tokens=10, chunk_size=4)
        assert len(ids) >= 8  # at least the prompt

    def test_kv_memory(self):
        model = self._make_model()
        mem = model.total_kv_memory_bytes()
        assert mem > 0

    def test_causal_masking(self):
        """Verify causal masking: token i should not attend to token i+1."""
        model = self._make_model()
        tokens = torch.tensor([[42, 100, 200]], dtype=torch.long)
        with torch.no_grad():
            _, attn_probs = model.forward_block(tokens, return_attn_probs=True)

        # For the first token (position 0), it should only attend to itself
        # attn_probs[layer]: [n_heads, 3, 3]
        probs = attn_probs[0]  # first layer
        # Token 0 attending to positions 1, 2 should be 0
        assert probs[:, 0, 1:].abs().max() < 1e-5, \
            "Token 0 should not attend to future positions"

    def test_deterministic_greedy(self):
        """Same input should give same output in greedy mode."""
        model = self._make_model()
        prompt = torch.tensor([[42, 100, 200]], dtype=torch.long)

        with torch.no_grad():
            ids1 = model.generate_greedy(prompt.clone(), max_new_tokens=5)
            ids2 = model.generate_greedy(prompt.clone(), max_new_tokens=5)

        assert ids1 == ids2, "Greedy generation should be deterministic"


class TestOPTModelRealConfig(object):
    """Tests with real OPT configs (125M-scale, CPU only)."""

    @pytest.mark.slow
    def test_opt125m_config_shapes(self):
        """Verify OPT-125M config produces correct shapes."""
        config = OPT_CONFIGS['opt-125m']
        model = OPTModel(config, dtype=torch.float16)
        model = model.to('cpu').to(torch.float16)
        model.lm_head_weight = model.embed_tokens.weight
        model.eval()
        model.init_kv_caches(device='cpu')

        token = torch.tensor([[2]], dtype=torch.long)  # BOS
        with torch.no_grad():
            logits, _ = model.forward_block(token)

        assert logits.shape == (1, 1, 50272), \
            "Expected (1, 1, 50272), got %s" % (logits.shape,)
