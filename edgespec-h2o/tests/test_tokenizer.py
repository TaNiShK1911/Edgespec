# -*- coding: utf-8 -*-
"""Tests for the vendored GPT-2 BPE tokenizer (G1.2).

Validates token IDs against HuggingFace golden references.
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_golden_path():
    """Find golden token IDs file."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Try both opt-125m and generic paths
    candidates = [
        os.path.join(base, 'data', 'golden', 'opt-125m', 'golden_token_ids.json'),
        os.path.join(base, 'data', 'golden', 'golden_token_ids.json'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _get_tokenizer():
    """Get tokenizer if vocab/merges exist."""
    from edgespec.config import VOCAB_FILE, MERGES_FILE
    if not os.path.exists(VOCAB_FILE) or not os.path.exists(MERGES_FILE):
        pytest.skip("Vocab/merges files not found — run export_golden.py first")
    from edgespec.tokenizer import get_tokenizer
    return get_tokenizer()


class TestTokenizerBasic(object):
    """Basic tokenizer tests (no golden file needed)."""

    def test_encode_returns_list(self):
        tokenizer = _get_tokenizer()
        result = tokenizer.encode("Hello")
        assert isinstance(result, list)
        assert all(isinstance(t, int) for t in result)

    def test_bos_token(self):
        tokenizer = _get_tokenizer()
        result = tokenizer.encode("Hello", add_bos=True)
        assert result[0] == 2, "BOS token should be </s> = 2"

    def test_no_bos(self):
        tokenizer = _get_tokenizer()
        result = tokenizer.encode("Hello", add_bos=False)
        # Should not start with BOS
        assert len(result) > 0

    def test_roundtrip(self):
        tokenizer = _get_tokenizer()
        text = "Hello, world! This is a test."
        ids = tokenizer.encode(text, add_bos=False)
        decoded = tokenizer.decode(ids)
        assert decoded == text, "Round-trip failed: %r != %r" % (decoded, text)

    def test_empty_string(self):
        tokenizer = _get_tokenizer()
        result = tokenizer.encode("", add_bos=True)
        assert result == [2], "Empty string should produce only BOS"

    def test_vocab_size(self):
        tokenizer = _get_tokenizer()
        assert tokenizer.vocab_size == 50272, \
            "OPT vocab size should be 50272, got %d" % tokenizer.vocab_size


class TestTokenizerGolden(object):
    """Validate against HuggingFace golden token IDs (G1.2)."""

    def test_golden_token_ids(self):
        golden_path = _get_golden_path()
        if golden_path is None:
            pytest.skip("Golden token IDs file not found — run export_golden.py first")

        tokenizer = _get_tokenizer()

        with open(golden_path, 'r', encoding='utf-8') as f:
            golden = json.load(f)

        n_pass = 0
        n_fail = 0
        failures = []

        for entry in golden:
            if 'error' in entry:
                continue
            text = entry['text']
            expected = entry['token_ids']

            actual = tokenizer.encode(text, add_bos=True)

            if actual == expected:
                n_pass += 1
            else:
                n_fail += 1
                failures.append({
                    'text': text[:50],
                    'expected_len': len(expected),
                    'actual_len': len(actual),
                    'first_diff_pos': next(
                        (i for i, (a, e) in enumerate(zip(actual, expected)) if a != e),
                        min(len(actual), len(expected))
                    ),
                })

        total = n_pass + n_fail
        pct = 100.0 * n_pass / total if total > 0 else 0

        if failures:
            print("\nToken ID mismatches:")
            for f in failures[:5]:
                print("  '%s...': expected %d tokens, got %d, first diff at pos %d" % (
                    f['text'], f['expected_len'], f['actual_len'], f['first_diff_pos']))

        assert n_pass == total, \
            "Token ID match: %d/%d (%.1f%%) — need 100%% for G1.2" % (
                n_pass, total, pct)
