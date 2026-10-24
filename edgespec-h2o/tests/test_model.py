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


class TestQuantizedInferenceAndLoader(object):
    """Tests for quantized model loading, QLinear integration, and perturbation."""

    def _create_quantized_tiny_state_dict(self):
        """Create a complete synthetic quantized state dict for TINY_CONFIG."""
        torch.manual_seed(42)
        d_model = TINY_CONFIG['d_model']
        d_ffn = TINY_CONFIG['d_ffn']
        vocab_size = TINY_CONFIG['vocab_size']
        max_pos = TINY_CONFIG['max_positions'] + TINY_CONFIG['position_offset']

        from edgespec.quant.qlinear import pack_int4, compute_scales_zeros

        sd = {
            'embed_tokens.weight': torch.randn(vocab_size, d_model, dtype=torch.float16),
            'embed_positions.weight': torch.randn(max_pos, d_model, dtype=torch.float16),
            'final_layer_norm.weight': torch.ones(d_model, dtype=torch.float16),
            'final_layer_norm.bias': torch.zeros(d_model, dtype=torch.float16),
        }

        for l_idx in range(TINY_CONFIG['n_layers']):
            sd['layers.%d.self_attn_layer_norm.weight' % l_idx] = torch.ones(d_model, dtype=torch.float16)
            sd['layers.%d.self_attn_layer_norm.bias' % l_idx] = torch.zeros(d_model, dtype=torch.float16)
            sd['layers.%d.final_layer_norm.weight' % l_idx] = torch.ones(d_model, dtype=torch.float16)
            sd['layers.%d.final_layer_norm.bias' % l_idx] = torch.zeros(d_model, dtype=torch.float16)

            for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                w = torch.randn(d_model, d_model, dtype=torch.float32)
                sc, zr = compute_scales_zeros(w, group_size=32)
                qw = pack_int4(w.half(), sc, zr, group_size=32)
                prefix = 'layers.%d.self_attn.%s' % (l_idx, proj)
                sd[prefix + '.qweight'] = qw
                sd[prefix + '.scales'] = sc
                sd[prefix + '.zeros'] = zr
                sd[prefix + '.bias'] = torch.zeros(d_model, dtype=torch.float16)

            # fc1
            w1 = torch.randn(d_ffn, d_model, dtype=torch.float32)
            sc1, zr1 = compute_scales_zeros(w1, group_size=32)
            qw1 = pack_int4(w1.half(), sc1, zr1, group_size=32)
            sd['layers.%d.fc1.qweight' % l_idx] = qw1
            sd['layers.%d.fc1.scales' % l_idx] = sc1
            sd['layers.%d.fc1.zeros' % l_idx] = zr1
            sd['layers.%d.fc1.bias' % l_idx] = torch.zeros(d_ffn, dtype=torch.float16)

            # fc2
            w2 = torch.randn(d_model, d_ffn, dtype=torch.float32)
            sc2, zr2 = compute_scales_zeros(w2, group_size=32)
            qw2 = pack_int4(w2.half(), sc2, zr2, group_size=32)
            sd['layers.%d.fc2.qweight' % l_idx] = qw2
            sd['layers.%d.fc2.scales' % l_idx] = sc2
            sd['layers.%d.fc2.zeros' % l_idx] = zr2
            sd['layers.%d.fc2.bias' % l_idx] = torch.zeros(d_model, dtype=torch.float16)

        return sd

    def test_quantized_loader_integration(self):
        import tempfile, shutil
        from edgespec.model.loader import export_model, load_model
        from edgespec.quant.qlinear import QLinear
        tmpdir = tempfile.mkdtemp()
        try:
            ckpt_path = os.path.join(tmpdir, "tiny_quant.edgespec")
            from edgespec.config import OPT_CONFIGS
            OPT_CONFIGS["tiny"] = TINY_CONFIG

            sd = self._create_quantized_tiny_state_dict()
            export_model(sd, "tiny", ckpt_path)

            model = load_model(ckpt_path, device="cpu", dtype=torch.float16, quantized=True)
            assert isinstance(model.layers[0].self_attn.q_proj, QLinear)
            assert isinstance(model.layers[0].fc1, QLinear)
            assert isinstance(model.layers[0].fc2, QLinear)

            inp = torch.tensor([[2, 42, 100]], dtype=torch.long)
            model.init_kv_caches(device="cpu")
            logits, _ = model.forward_block(inp)
            assert logits.shape == (1, 3, TINY_CONFIG["vocab_size"])
        finally:
            shutil.rmtree(tmpdir)
            from edgespec.config import OPT_CONFIGS
            OPT_CONFIGS.pop("tiny", None)

    def test_quantized_weight_perturbation_changes_model_logits(self):
        import tempfile, shutil
        from edgespec.model.loader import export_model, load_model
        tmpdir = tempfile.mkdtemp()
        try:
            ckpt_path = os.path.join(tmpdir, "tiny_quant.edgespec")
            from edgespec.config import OPT_CONFIGS
            OPT_CONFIGS["tiny"] = TINY_CONFIG

            sd = self._create_quantized_tiny_state_dict()
            export_model(sd, "tiny", ckpt_path)

            model = load_model(ckpt_path, device="cpu", dtype=torch.float16, quantized=True)
            model.init_kv_caches(device="cpu")
            inp = torch.tensor([[2, 10, 20]], dtype=torch.long)
            with torch.no_grad():
                orig_logits, _ = model.forward_block(inp)
                orig_logits = orig_logits.clone()

            # Perturb quantized weight byte in fc1
            ql = model.layers[0].fc1
            ql.qweight.data[0, 0] = (ql.qweight[0, 0] ^ 0xFF) & 0xFF

            model.reset_caches()
            with torch.no_grad():
                pert_logits, _ = model.forward_block(inp)

            diff = (pert_logits - orig_logits).abs().max().item()
            assert diff > 1e-4, "Modifying quantized weight in QLinear must alter model logits"
        finally:
            shutil.rmtree(tmpdir)
            from edgespec.config import OPT_CONFIGS
            OPT_CONFIGS.pop("tiny", None)

    def test_checkpoint_reload_reproduces_inference(self):
        import tempfile, shutil
        from edgespec.model.loader import export_model, load_model
        tmpdir = tempfile.mkdtemp()
        try:
            ckpt_path = os.path.join(tmpdir, "tiny_quant.edgespec")
            from edgespec.config import OPT_CONFIGS
            OPT_CONFIGS["tiny"] = TINY_CONFIG

            sd = self._create_quantized_tiny_state_dict()
            export_model(sd, "tiny", ckpt_path)

            model1 = load_model(ckpt_path, device="cpu", dtype=torch.float16, quantized=True)
            model2 = load_model(ckpt_path, device="cpu", dtype=torch.float16, quantized=True)

            model1.init_kv_caches(device="cpu")
            model2.init_kv_caches(device="cpu")
            inp = torch.tensor([[2, 15, 25, 35]], dtype=torch.long)

            with torch.no_grad():
                out1, _ = model1.forward_block(inp)
                out2, _ = model2.forward_block(inp)

            diff = (out1 - out2).abs().max().item()
            assert diff == 0.0, "Reloading checkpoint must reproduce inference bit-for-bit"
        finally:
            shutil.rmtree(tmpdir)
            from edgespec.config import OPT_CONFIGS
            OPT_CONFIGS.pop("tiny", None)

    def test_missing_tensor_raises_error(self):
        import tempfile, shutil
        from edgespec.model.loader import export_model, load_model
        tmpdir = tempfile.mkdtemp()
        try:
            ckpt_path = os.path.join(tmpdir, "corrupt.edgespec")
            from edgespec.config import OPT_CONFIGS
            OPT_CONFIGS["tiny"] = TINY_CONFIG

            sd = self._create_quantized_tiny_state_dict()
            del sd['layers.0.self_attn.q_proj.scales']
            export_model(sd, "tiny", ckpt_path)

            with pytest.raises(ValueError):
                load_model(ckpt_path, device="cpu", dtype=torch.float16, quantized=True)
        finally:
            shutil.rmtree(tmpdir)
            from edgespec.config import OPT_CONFIGS
            OPT_CONFIGS.pop("tiny", None)

    def test_unknown_tensor_raises_error(self):
        import tempfile, shutil
        from edgespec.model.loader import export_model, load_model
        tmpdir = tempfile.mkdtemp()
        try:
            ckpt_path = os.path.join(tmpdir, "unknown.edgespec")
            from edgespec.config import OPT_CONFIGS
            OPT_CONFIGS["tiny"] = TINY_CONFIG

            sd = self._create_quantized_tiny_state_dict()
            sd['random_unknown_tensor.weight'] = torch.randn(10, 10, dtype=torch.float16)
            export_model(sd, "tiny", ckpt_path)

            with pytest.raises(KeyError):
                load_model(ckpt_path, device="cpu", dtype=torch.float16, quantized=True)
        finally:
            shutil.rmtree(tmpdir)
            from edgespec.config import OPT_CONFIGS
            OPT_CONFIGS.pop("tiny", None)


class TestGoldenLogitsParity(object):
    """G1.3: FP32 reference logits parity against Hugging Face reference."""

    def test_fp32_golden_logits_top1_and_diff(self):
        import json
        golden_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'data', 'golden', 'opt-125m', 'golden_logits.json')
        if not os.path.exists(golden_file):
            golden_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       'data', 'golden', 'golden_logits.json')
        if not os.path.exists(golden_file):
            pytest.skip("Golden logits file not found — run export_golden.py first")

        try:
            from edgespec.model.loader import load_model_from_hf
            model = load_model_from_hf("facebook/opt-125m", device="cpu", dtype=torch.float32)
        except (ImportError, ModuleNotFoundError):
            # Check local checkpoint path (e.g. on Jetson Nano)
            local_bin = os.path.expanduser("~/edgespec/models/opt-125m/pytorch_model.bin")
            if not os.path.exists(local_bin):
                # also check relative data/weights
                local_bin = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        'data', 'weights', 'opt-125m', 'pytorch_model.bin')
            if os.path.exists(local_bin):
                from edgespec.config import OPT_CONFIGS
                def assign_p(mod, full_name, t):
                    parts = full_name.split('.')
                    obj = mod
                    for p in parts[:-1]:
                        if p.isdigit():
                            obj = obj[int(p)]
                        else:
                            obj = getattr(obj, p)
                    param = getattr(obj, parts[-1])
                    if isinstance(param, torch.nn.Parameter):
                        param.data.copy_(t)
                    else:
                        setattr(obj, parts[-1], t)

                config = OPT_CONFIGS['opt-125m']
                model = OPTModel(config, dtype=torch.float32)
                sd = torch.load(local_bin, map_location="cpu")
                mapping = {}
                mapping["embed_tokens.weight"] = sd.get("model.decoder.embed_tokens.weight")
                mapping["embed_positions.weight"] = sd.get("model.decoder.embed_positions.weight")
                if "model.decoder.final_layer_norm.weight" in sd:
                    mapping["final_layer_norm.weight"] = sd["model.decoder.final_layer_norm.weight"]
                    mapping["final_layer_norm.bias"] = sd["model.decoder.final_layer_norm.bias"]
                for i in range(config["n_layers"]):
                    hf_pfx = "model.decoder.layers.%d" % i
                    our_pfx = "layers.%d" % i
                    for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                        mapping["%s.self_attn.%s.weight" % (our_pfx, proj)] = sd.get("%s.self_attn.%s.weight" % (hf_pfx, proj))
                        mapping["%s.self_attn.%s.bias" % (our_pfx, proj)] = sd.get("%s.self_attn.%s.bias" % (hf_pfx, proj))
                    for ln in ("self_attn_layer_norm", "final_layer_norm"):
                        mapping["%s.%s.weight" % (our_pfx, ln)] = sd.get("%s.%s.weight" % (hf_pfx, ln))
                        mapping["%s.%s.bias" % (our_pfx, ln)] = sd.get("%s.%s.bias" % (hf_pfx, ln))
                    mapping["%s.mlp.fc1.weight" % our_pfx] = sd.get("%s.fc1.weight" % hf_pfx)
                    mapping["%s.mlp.fc1.bias" % our_pfx] = sd.get("%s.fc1.bias" % hf_pfx)
                    mapping["%s.mlp.fc2.weight" % our_pfx] = sd.get("%s.fc2.weight" % hf_pfx)
                    mapping["%s.mlp.fc2.bias" % our_pfx] = sd.get("%s.fc2.bias" % hf_pfx)
                for name, tensor in mapping.items():
                    if tensor is not None:
                        assign_p(model, name, tensor.to(torch.float32))
                model.lm_head_weight = model.embed_tokens.weight
                model.eval()
            else:
                pytest.skip("transformers not installed and local weights not found")

        with open(golden_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_positions = 0
        matches = 0
        diffs = []

        for p_data in data['prompts']:
            inp = torch.tensor([p_data['input_ids']], dtype=torch.long)
            model.reset_caches()
            model.init_kv_caches(max_seq_len=max(2048, inp.shape[1] + 16), device="cpu")
            with torch.no_grad():
                logits, _ = model.forward_block(inp)

            pred_top1 = logits[0].argmax(dim=-1).tolist()
            expected_top1 = p_data['top1_ids']

            for got_id, exp_id in zip(pred_top1, expected_top1):
                if got_id == exp_id:
                    matches += 1
                total_positions += 1

            last_logits = logits[0, -1, :]
            expected_top20_ids = p_data['last_pos_logits_top20_ids']
            expected_top20_vals = p_data['last_pos_logits_top20_vals']
            actual_vals = last_logits[expected_top20_ids].tolist()

            for av, ev in zip(actual_vals, expected_top20_vals):
                diffs.append(abs(av - ev))

        agreement_pct = (matches / total_positions) * 100.0 if total_positions > 0 else 0.0
        mean_diff = sum(diffs) / len(diffs) if diffs else 0.0

        print("\nGolden Logits: %d/%d top-1 (%.2f%%), mean diff = %.6f" % (
            matches, total_positions, agreement_pct, mean_diff))

        assert total_positions >= 500, "Must evaluate over >= 500 positions (got %d)" % total_positions
        assert agreement_pct >= 98.0, "Top-1 agreement %.2f%% < 98%%" % agreement_pct
        assert mean_diff <= 0.05, "Mean abs logit diff %.4f > 0.05" % mean_diff

