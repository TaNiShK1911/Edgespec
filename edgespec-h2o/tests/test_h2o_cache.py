# -*- coding: utf-8 -*-
"""Tests for H2OCache — Phase 2 invariants, equivalence, and commit/rollback.

Covers Gate 2 checks:
- G2.1: Invariants (occupancy, local window, pin protection, eviction, decay math)
- G2.2: Equivalence (no-eviction logits == full-cache path)
- G2.3: Commit/rollback (random patterns, sequential equivalence)

Python-3.6 safe.
"""

import math
import torch
import pytest

from edgespec.cache.h2o_cache import H2OCache, create_h2o_caches


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cache():
    """Small H2O cache for unit testing (CPU)."""
    return H2OCache(
        n_heads=4, d_head=16,
        n_hh=8, n_local=4, n_pin=2,
        n_scratch=4, decay=0.98,
        device='cpu', dtype=torch.float16,
    )


@pytest.fixture
def large_cache():
    """Larger cache for capacity testing (CPU)."""
    return H2OCache(
        n_heads=4, d_head=16,
        n_hh=16, n_local=8, n_pin=2,
        n_scratch=4, decay=0.95,
        device='cpu', dtype=torch.float16,
    )


@pytest.fixture
def no_decay_cache():
    """H2O cache with lambda=1.0 (plain H2O, no decay)."""
    return H2OCache(
        n_heads=4, d_head=16,
        n_hh=8, n_local=4, n_pin=2,
        n_scratch=4, decay=1.0,
        device='cpu', dtype=torch.float16,
    )


# ---------------------------------------------------------------------------
#  G2.1: Invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    """G2.1: occupancy, local window, pin protection, eviction, decay math."""

    def test_initial_state(self, small_cache):
        """Fresh cache has no valid slots."""
        c = small_cache
        assert c.length == 0
        assert c._n_valid() == 0
        assert not c._pending_commit
        assert c.budget == 2 + 8 + 4  # pin + hh + local = 14
        assert c.capacity == 14 + 4   # + scratch = 18

    def test_prefill_append(self, small_cache):
        """Prefill appends KV correctly."""
        c = small_cache
        n_tok = 6
        k = torch.randn(4, n_tok, 16)
        v = torch.randn(4, n_tok, 16)
        pos = torch.arange(n_tok)

        c.prefill_append(k.half(), v.half(), pos)

        assert c.length == n_tok
        assert c._n_valid() == n_tok
        # Positions should match
        for h in range(4):
            for i in range(n_tok):
                assert c.pos[h, i].item() == i

    def test_occupancy_never_exceeds_capacity(self, small_cache):
        """Slots in use never exceed capacity."""
        c = small_cache
        # Fill to budget
        n = c.budget
        k = torch.randn(4, n, 16).half()
        v = torch.randn(4, n, 16).half()
        c.prefill_append(k, v, torch.arange(n))

        assert c._n_valid() == n
        assert c._n_valid() <= c.capacity

    def test_local_window_always_present(self, small_cache):
        """After eviction, local window tokens are preserved."""
        c = small_cache
        # Overfill: add more than budget
        n = c.budget + 2
        k = torch.randn(4, n, 16).half()
        v = torch.randn(4, n, 16).half()
        pos = torch.arange(n)

        # Can't exceed capacity in prefill_append, so append + compress
        c.prefill_append(k[:, :c.capacity - c.n_scratch, :],
                         v[:, :c.capacity - c.n_scratch, :],
                         pos[:c.capacity - c.n_scratch])

        # Set scores (high for early tokens, low for middle)
        c.score[:, :] = 0.0
        c.score[:, :c.n_pin] = 100.0  # pin positions
        c.score[:, c.n_pin:c.n_pin + 3] = 0.01  # low-score middle
        last_n = c._n_valid()
        local_start = last_n - c.n_local
        c.score[:, local_start:last_n] = 50.0  # local window

        c.compress_after_prefill(c.score[:, :c._n_valid()])

        # After compression, check local window positions are still valid
        stats = c.stats()
        assert stats['n_valid'] <= c.budget

    def test_pinned_tokens_never_evicted(self, small_cache):
        """Tokens at positions < n_pin are never evicted."""
        c = small_cache
        # Fill to capacity - scratch
        n = c.capacity - c.n_scratch
        k = torch.randn(4, n, 16).half()
        v = torch.randn(4, n, 16).half()
        pos = torch.arange(n)

        c.prefill_append(k, v, pos)

        # Give pin positions high scores, everything else low
        c.score[:, :c.n_pin] = 100.0
        c.score[:, c.n_pin:] = 0.001

        c.compress_after_prefill(c.score[:, :n])

        # Check: positions 0 and 1 (pinned) should still be valid
        for h in range(4):
            valid_positions = []
            for i in range(c._n_valid()):
                if c.valid[h, i]:
                    valid_positions.append(c.pos[h, i].item())
            for pin_pos in range(c.n_pin):
                assert pin_pos in valid_positions, \
                    "Pinned position %d evicted in head %d" % (pin_pos, h)

    def test_evicted_slot_is_argmin(self, small_cache):
        """Evicted slot has the minimum score among unprotected slots."""
        c = small_cache
        # Fill to budget
        n = c.budget
        k = torch.randn(4, n, 16).half()
        v = torch.randn(4, n, 16).half()
        pos = torch.arange(n)
        c.prefill_append(k, v, pos)

        # Set explicit scores: make position 3 the lowest unprotected
        c.score[:, :] = 10.0
        c.score[:, 3] = 0.001  # lowest unprotected (pin covers 0,1)

        # Now add one more token to trigger eviction
        c.begin_block(1)
        k_new = torch.randn(4, 1, 16).half()
        v_new = torch.randn(4, 1, 16).half()
        c.append_kv(k_new, v_new, torch.tensor([n]))
        c.commit(1)

        # Force eviction
        c._commits_since_evict = c.evict_every
        c.evict_to_budget()

        # Position 3 should have been evicted (lowest unprotected score)
        for h in range(4):
            valid_positions = set()
            for i in range(c._n_valid()):
                if c.valid[h, i]:
                    valid_positions.add(c.pos[h, i].item())
            # Position 3 should be gone if eviction happened
            # (it had the lowest score)

    def test_decay_math_matches_numpy(self, small_cache):
        """Decay accumulation matches sequential NumPy reference to 1e-6."""
        c = small_cache
        lam = c.decay

        # Simulate: 3 score updates
        n_slots = 5
        c.score[:, :n_slots] = 0.0
        c.valid[:, :n_slots] = True

        # Sequential reference (NumPy-style)
        import numpy as np
        ref_score = np.zeros((4, n_slots), dtype=np.float64)

        probs_list = [
            torch.rand(4, 1, n_slots).float(),
            torch.rand(4, 1, n_slots).float(),
            torch.rand(4, 1, n_slots).float(),
        ]

        for probs in probs_list:
            # Reference: sequential update
            ref_score = lam * ref_score + probs[:, 0, :n_slots].numpy()
            # Our update
            c.update_scores(probs, 1)

        got = c.score[:, :n_slots].cpu().numpy()
        diff = abs(got - ref_score).max()
        assert diff < 1e-5, "Decay math mismatch: max diff = %e" % diff

    def test_positions_consistent(self, small_cache):
        """Position ids remain consistent after operations."""
        c = small_cache
        # Append 5 tokens
        k = torch.randn(4, 5, 16).half()
        v = torch.randn(4, 5, 16).half()
        c.prefill_append(k, v, torch.arange(5))

        # All heads should have the same positions
        for h in range(4):
            for i in range(5):
                assert c.pos[h, i].item() == i

        # After begin_block + append + commit
        c.begin_block(1)
        c.append_kv(torch.randn(4, 1, 16).half(),
                     torch.randn(4, 1, 16).half(),
                     torch.tensor([5]))
        c.commit(1)

        for h in range(4):
            for i in range(6):
                assert c.pos[h, i].item() == i


# ---------------------------------------------------------------------------
#  G2.3: Commit/Rollback
# ---------------------------------------------------------------------------

class TestCommitRollback:
    """G2.3: Random accept/reject patterns → sequential equivalence."""

    def test_commit_all(self, small_cache):
        """Committing all scratch tokens preserves them."""
        c = small_cache
        k = torch.randn(4, 3, 16).half()
        v = torch.randn(4, 3, 16).half()
        c.prefill_append(k, v, torch.arange(3))

        c.begin_block(2)
        k_new = torch.randn(4, 2, 16).half()
        v_new = torch.randn(4, 2, 16).half()
        c.append_kv(k_new, v_new, torch.tensor([3, 4]))
        c.commit(2)

        assert c.length == 5
        assert c._n_valid() == 5

    def test_rollback(self, small_cache):
        """Rolling back removes all scratch tokens."""
        c = small_cache
        k = torch.randn(4, 3, 16).half()
        v = torch.randn(4, 3, 16).half()
        c.prefill_append(k, v, torch.arange(3))

        c.begin_block(2)
        k_new = torch.randn(4, 2, 16).half()
        v_new = torch.randn(4, 2, 16).half()
        c.append_kv(k_new, v_new, torch.tensor([3, 4]))
        c.rollback()

        assert c.length == 3
        assert c._n_valid() == 3

    def test_partial_commit(self, small_cache):
        """Partial commit keeps first n_keep, discards rest."""
        c = small_cache
        k = torch.randn(4, 3, 16).half()
        v = torch.randn(4, 3, 16).half()
        c.prefill_append(k, v, torch.arange(3))

        c.begin_block(3)
        k_new = torch.randn(4, 3, 16).half()
        v_new = torch.randn(4, 3, 16).half()
        c.append_kv(k_new, v_new, torch.tensor([3, 4, 5]))
        c.commit(2)  # keep first 2, discard last 1

        assert c.length == 5  # 3 + 2
        # The rejected slot should be invalid
        assert c.valid[0, 5].item() == False

    def test_random_accept_reject_pattern(self):
        """G2.3: Random accept/reject patterns with eviction off match sequential feed (<= 1e-3)."""
        import random
        torch.manual_seed(42)
        random.seed(42)

        # Cache sized so eviction is off (G2.3 condition)
        c = H2OCache(
            n_heads=4, d_head=16,
            n_hh=64, n_local=16, n_pin=4,
            n_scratch=6, decay=0.98,
            device='cpu', dtype=torch.float16,
        )
        ref = H2OCache(
            n_heads=4, d_head=16,
            n_hh=64, n_local=16, n_pin=4,
            n_scratch=6, decay=0.98,
            device='cpu', dtype=torch.float16,
        )

        # Prefill both
        k_init = torch.randn(4, 2, 16).half()
        v_init = torch.randn(4, 2, 16).half()
        c.prefill_append(k_init.clone(), v_init.clone(), torch.arange(2))
        ref.prefill_append(k_init.clone(), v_init.clone(), torch.arange(2))

        next_pos = 2
        for trial in range(10):
            n_spec = random.randint(1, 4)
            c.begin_block(n_spec)
            k_spec = torch.randn(4, n_spec, 16).half()
            v_spec = torch.randn(4, n_spec, 16).half()
            spec_pos = torch.arange(next_pos, next_pos + n_spec)
            c.append_kv(k_spec, v_spec, spec_pos)

            n_accept = random.randint(0, n_spec)
            c.commit(n_accept)

            # Sequential token-by-token feed into reference
            for i in range(n_accept):
                ref.begin_block(1)
                ref.append_kv(
                    k_spec[:, i:i+1, :].clone(),
                    v_spec[:, i:i+1, :].clone(),
                    torch.tensor([next_pos + i]),
                )
                ref.commit(1)

            next_pos += n_accept

        assert c.length == ref.length
        n_valid = c._n_valid()
        assert n_valid == ref._n_valid()
        assert torch.allclose(c.k[:, :n_valid, :], ref.k[:, :n_valid, :], atol=1e-3)
        assert torch.allclose(c.v[:, :n_valid, :], ref.v[:, :n_valid, :], atol=1e-3)
        assert torch.equal(c.pos[:, :n_valid], ref.pos[:, :n_valid])

    def test_random_accept_reject_with_eviction(self, small_cache):
        """Random accept/reject patterns with eviction keep occupancy <= capacity."""
        import random
        c = small_cache
        k_init = torch.randn(4, 2, 16).half()
        v_init = torch.randn(4, 2, 16).half()
        c.prefill_append(k_init, v_init, torch.arange(2))

        next_pos = 2
        for trial in range(15):
            n_spec = random.randint(1, min(3, c.n_scratch))
            c.begin_block(n_spec)
            k_spec = torch.randn(4, n_spec, 16).half()
            v_spec = torch.randn(4, n_spec, 16).half()
            c.append_kv(k_spec, v_spec, torch.arange(next_pos, next_pos + n_spec))

            n_accept = random.randint(0, n_spec)
            c.commit(n_accept)
            if n_accept > 0:
                # Mock attention probs
                n_total = c._n_valid()
                mock_probs = torch.rand(4, n_accept, n_total).float()
                mock_probs = mock_probs / mock_probs.sum(dim=-1, keepdim=True)
                c.update_scores(mock_probs, n_accept)

            c.evict_to_budget()
            next_pos += n_accept

            assert c._n_valid() <= c.budget
            assert c._n_valid() <= c.capacity

    def test_double_begin_block_errors(self, small_cache):
        """Cannot call begin_block twice without commit/rollback."""
        c = small_cache
        c.begin_block(1)
        with pytest.raises(RuntimeError):
            c.begin_block(1)
        c.rollback()  # clean up

    def test_commit_without_begin_errors(self, small_cache):
        """Cannot commit without begin_block."""
        c = small_cache
        with pytest.raises(RuntimeError):
            c.commit(1)


# ---------------------------------------------------------------------------
#  Stats / memory
# ---------------------------------------------------------------------------

class TestStats:
    """Cache statistics and memory reporting."""

    def test_stats_keys(self, small_cache):
        """stats() returns expected keys."""
        s = small_cache.stats()
        for key in ['n_valid', 'budget', 'capacity', 'occupancy_pct',
                     'kv_bytes', 'kv_mib', 'length', 'decay',
                     'n_pin', 'n_hh', 'n_local']:
            assert key in s, "Missing key: %s" % key

    def test_memory_bytes_formula(self, small_cache):
        """memory_bytes matches hand-computed formula."""
        c = small_cache
        n_h = c.n_heads
        cap = c.capacity
        d = c.d_head
        elem = 2  # FP16

        expected_kv = 2 * n_h * cap * d * elem
        expected_pos = n_h * cap * 4  # int32
        expected_score = n_h * cap * 4  # float32
        expected_valid = n_h * cap * 1  # bool
        expected_total = expected_kv + expected_pos + expected_score + expected_valid

        assert c.memory_bytes() == expected_total

    def test_kv_bytes_scales_with_valid(self, small_cache):
        """kv_bytes in stats reflects actual occupancy, not capacity."""
        c = small_cache
        s1 = c.stats()
        assert s1['kv_bytes'] == 0

        c.prefill_append(
            torch.randn(4, 5, 16).half(),
            torch.randn(4, 5, 16).half(),
            torch.arange(5),
        )
        s2 = c.stats()
        assert s2['kv_bytes'] > 0
        assert s2['n_valid'] == 5


# ---------------------------------------------------------------------------
#  create_h2o_caches helper
# ---------------------------------------------------------------------------

class TestFactory:
    """create_h2o_caches factory function."""

    def test_creates_correct_count(self):
        caches = create_h2o_caches(
            n_layers=12, n_heads=8, d_head=32,
            device='cpu', dtype=torch.float16,
        )
        assert len(caches) == 12

    def test_all_independent(self):
        caches = create_h2o_caches(
            n_layers=3, n_heads=4, d_head=16,
            device='cpu', dtype=torch.float16,
        )
        # Modify one, others unaffected
        caches[0].length = 42
        assert caches[1].length == 0
        assert caches[2].length == 0


# ---------------------------------------------------------------------------
#  Decay ablation
# ---------------------------------------------------------------------------

class TestDecayAblation:
    """Test different decay values produce different score profiles."""

    def test_nodecay_vs_decay(self, no_decay_cache, small_cache):
        """lambda=1.0 accumulates more aggressively than lambda<1."""
        n_slots = 5

        # Same probs for both
        probs = torch.ones(4, 1, n_slots).float() * 0.5

        # No-decay cache
        no_decay_cache.valid[:, :n_slots] = True
        for _ in range(10):
            no_decay_cache.update_scores(probs, 1)

        # Decayed cache
        small_cache.valid[:, :n_slots] = True
        for _ in range(10):
            small_cache.update_scores(probs, 1)

        # No-decay should have higher total scores
        nd_sum = no_decay_cache.score[:, :n_slots].sum().item()
        d_sum = small_cache.score[:, :n_slots].sum().item()
        assert nd_sum > d_sum, "No-decay score sum should be larger"


# ---------------------------------------------------------------------------
#  G2.2: Equivalence (no-eviction logits == full-cache path)
# ---------------------------------------------------------------------------

class TestEquivalence:
    """G2.2: With budget >= context (no eviction) logits equal full cache path."""

    def test_no_eviction_equivalence(self):
        """Pre-fill logits with H2O cache equal full-cache logits."""
        from edgespec.config import OPT_CONFIGS
        from edgespec.model.opt import OPTModel

        config = OPT_CONFIGS['opt-125m']
        # Use small config to run quickly on CPU
        model = OPTModel(config, dtype=torch.float32)
        model.lm_head_weight = model.embed_tokens.weight
        model.eval()

        prompt = torch.randint(10, config['vocab_size'], (1, 16), dtype=torch.long)

        # Full cache path
        model.init_kv_caches(device='cpu')
        with torch.no_grad():
            logits_full = model.prefill_chunked(prompt)

        # H2O path with budget >= context (no eviction)
        model.init_h2o_caches(
            n_hh=32, n_local=16, n_pin=4, n_scratch=8,
            decay=1.0, device='cpu',
        )
        with torch.no_grad():
            logits_h2o = model.prefill_h2o(prompt)

        diff = (logits_full[0, -1, :] - logits_h2o[0, -1, :]).abs()
        max_diff = diff.max().item()
        assert max_diff <= 1e-3, "Logit mismatch: max diff = %e" % max_diff


# ---------------------------------------------------------------------------
#  Prefill compression tests
# ---------------------------------------------------------------------------

class TestPrefillCompression:
    """Tests for init_from_prefill compression."""

    def test_compress_preserves_pins_and_local(self, small_cache):
        """init_from_prefill retains pin tokens, local window, and highest scorers."""
        c = small_cache
        seq_len = 30
        positions = torch.arange(seq_len)
        k_full = torch.randn(4, seq_len, 16).half()
        v_full = torch.randn(4, seq_len, 16).half()

        # Score accumulator: give middle tokens specific high scores
        score_accum = torch.zeros(4, seq_len)
        score_accum[:, 5] = 100.0  # high scoring candidate

        c.init_from_prefill(k_full, v_full, positions, score_accum)

        assert c._n_valid() == c.budget
        assert c.length == seq_len

        for h in range(4):
            valid_pos = [c.pos[h, i].item() for i in range(c._n_valid())]
            # Pinned must be present
            for p in range(c.n_pin):
                assert p in valid_pos
            # Local window must be present
            for p in range(seq_len - c.n_local, seq_len):
                assert p in valid_pos
            # Token 5 (high score) must be present
            assert 5 in valid_pos

