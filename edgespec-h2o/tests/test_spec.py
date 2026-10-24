# -*- coding: utf-8 -*-
"""Tests for Phase 3 — speculative decoding components.

Covers:
- G3.1: Sampler math (toy distribution chi-square test)
- Drafter: n-gram index, suffix matching, importance weighting
- Gamma controller: efficiency computation, EMA updates
- Verifier integration: correctness checks

Python-3.6 safe (no walrus, no f-string=, etc.).
"""

import math
import random
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgespec.spec.drafter import SuffixDrafter, NgramIndex
from edgespec.spec.sampler import SpeculativeSampler
from edgespec.spec.gamma_ctrl import GammaController


# =========================================================================== #
#  Drafter tests (P3-1)
# =========================================================================== #

class TestNgramIndex(object):
    """Test the incremental n-gram index."""

    def test_basic_lookup(self):
        idx = NgramIndex(max_n=3)
        history = [10, 20, 30, 10, 20, 40]
        for i in range(len(history)):
            idx.add_token(history, i)

        # Unigram: token 10 appears at positions 0 and 3
        assert 0 in idx.lookup((10,), 1)
        assert 3 in idx.lookup((10,), 1)

        # Bigram: (10, 20) at positions 1 and 4
        assert 1 in idx.lookup((10, 20), 2)
        assert 4 in idx.lookup((10, 20), 2)

        # Trigram: (10, 20, 30) only at position 2
        assert 2 in idx.lookup((10, 20, 30), 3)
        assert len(idx.lookup((10, 20, 30), 3)) == 1

    def test_empty_lookup(self):
        idx = NgramIndex(max_n=4)
        assert idx.lookup((999,), 1) == []

    def test_reset(self):
        idx = NgramIndex(max_n=2)
        history = [1, 2, 3]
        for i in range(len(history)):
            idx.add_token(history, i)
        assert len(idx.lookup((1,), 1)) > 0
        idx.reset()
        assert len(idx.lookup((1,), 1)) == 0


class TestSuffixDrafter(object):
    """Test the suffix-match drafter."""

    def _make_drafter(self, selector='importance'):
        return SuffixDrafter(max_n=4, max_history=2048, selector=selector)

    def test_no_history_returns_empty(self):
        d = self._make_drafter()
        ids, probs, ml = d.draft(4)
        assert ids == []
        assert probs == []
        assert ml == 0

    def test_gamma_zero_returns_empty(self):
        d = self._make_drafter()
        d.commit_tokens([1, 2, 3, 1, 2, 4])
        ids, probs, ml = d.draft(0)
        assert ids == []

    def test_simple_repeat_pattern(self):
        d = self._make_drafter()
        # Pattern: [1, 2, 3, 1, 2, 3, 1, 2] — suffix [1, 2] should match
        d.commit_tokens([1, 2, 3, 1, 2, 3, 1, 2])
        ids, probs, ml = d.draft(3)
        # After suffix [1, 2], the continuation should be [3]
        assert len(ids) >= 1
        assert ids[0] == 3  # continuation after [1, 2] is [3]
        assert ml >= 2  # matched on bigram at least

    def test_importance_weighting(self):
        d = self._make_drafter(selector='importance')
        # Two occurrences of pattern [1, 2], one with high importance
        tokens = [1, 2, 10, 1, 2, 20, 1, 2]
        # Give position 2 (continuation 10) low importance
        # Give position 5 (continuation 20) high importance
        importance = [1.0, 1.0, 0.1, 1.0, 1.0, 10.0, 1.0, 1.0]
        d.commit_tokens(tokens, importance)

        ids, probs, ml = d.draft(1)
        # With importance weighting, token 20 should win
        assert len(ids) == 1
        assert ids[0] == 20  # higher importance

    def test_most_recent_selector(self):
        d = self._make_drafter(selector='most_recent')
        d.commit_tokens([1, 2, 10, 1, 2, 20, 1, 2])
        ids, probs, ml = d.draft(1)
        # Most recent match for [1, 2] is at position 4-5, continuation is 20
        assert ids[0] == 20

    def test_draft_probs_sum_to_one(self):
        d = self._make_drafter()
        d.commit_tokens([1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2])
        ids, probs, ml = d.draft(1)
        if probs:
            total = sum(probs[0].values())
            assert abs(total - 1.0) < 1e-6

    def test_reset_clears_state(self):
        d = self._make_drafter()
        d.commit_tokens([1, 2, 3, 1, 2])
        d.reset()
        ids, probs, ml = d.draft(3)
        assert ids == []


# =========================================================================== #
#  Sampler tests (P3-2 / G3.1)
# =========================================================================== #

class TestSpeculativeSampler(object):
    """Test speculative acceptance sampling."""

    def test_greedy_accept_exact_match(self):
        """Greedy mode: accept when draft == argmax(target)."""
        sampler = SpeculativeSampler(mode='greedy', lenience=1.0)
        # Target logits where token 5 is the argmax
        logits = torch.zeros(2, 10)  # gamma=1, vocab=10
        logits[0, 5] = 10.0  # strong preference for token 5
        logits[1, 3] = 10.0  # bonus position

        accepted, n_acc, mask = sampler.verify(
            [5], [{5: 1.0}], logits
        )
        assert n_acc == 1
        assert accepted[0] == 5
        assert mask == [True]
        # Bonus token
        assert accepted[1] == 3

    def test_greedy_reject_wrong_token(self):
        """Greedy mode: reject when draft != argmax(target)."""
        sampler = SpeculativeSampler(mode='greedy', lenience=1.0)
        logits = torch.zeros(2, 10)
        logits[0, 5] = 10.0
        logits[1, 3] = 10.0

        accepted, n_acc, mask = sampler.verify(
            [7], [{7: 1.0}], logits  # draft 7, target wants 5
        )
        assert n_acc == 0
        assert accepted[0] == 5  # correction to argmax
        assert mask == [False]

    def test_greedy_lenience_accepts_near_top(self):
        """Greedy with lenience: accept if p(x_hat) >= l * max(p)."""
        sampler = SpeculativeSampler(mode='greedy', lenience=0.5)
        logits = torch.zeros(2, 10)
        logits[0, 5] = 2.0   # highest
        logits[0, 7] = 1.5   # close to top
        logits[1, 0] = 1.0

        accepted, n_acc, mask = sampler.verify(
            [7], [{7: 1.0}], logits
        )
        # p(7) should be >= 0.5 * p(5) given these logits
        # After softmax, p(5) and p(7) will be close
        # This should accept with lenience 0.5
        assert n_acc == 1

    def test_greedy_multi_draft(self):
        """Greedy mode: verify multiple draft positions."""
        sampler = SpeculativeSampler(mode='greedy', lenience=1.0)
        # 3 drafts + 1 bonus = 4 logit positions
        logits = torch.zeros(4, 10)
        logits[0, 2] = 10.0
        logits[1, 3] = 10.0
        logits[2, 4] = 10.0
        logits[3, 5] = 10.0  # bonus

        accepted, n_acc, mask = sampler.verify(
            [2, 3, 4],
            [{2: 1.0}, {3: 1.0}, {4: 1.0}],
            logits
        )
        assert n_acc == 3
        assert accepted == [2, 3, 4, 5]  # all accepted + bonus

    def test_greedy_partial_accept(self):
        """Greedy mode: accept first draft, reject second."""
        sampler = SpeculativeSampler(mode='greedy', lenience=1.0)
        logits = torch.zeros(4, 10)
        logits[0, 2] = 10.0
        logits[1, 9] = 10.0  # wants 9, not 3
        logits[2, 4] = 10.0
        logits[3, 5] = 10.0

        accepted, n_acc, mask = sampler.verify(
            [2, 3, 4],
            [{2: 1.0}, {3: 1.0}, {4: 1.0}],
            logits
        )
        assert n_acc == 1  # only first accepted
        assert accepted[0] == 2
        assert accepted[1] == 9  # correction
        assert mask[0] is True
        assert mask[1] is False

    def test_sampling_acceptance_distribution(self):
        """G3.1: Statistical test — sampling from speculative sampler
        should match target distribution.

        Uses a simple target where p = [0.5, 0.3, 0.2] and uniform draft q.
        Over 10000 trials, chi-square test should show p > 0.01.
        """
        random.seed(42)
        torch.manual_seed(42)

        vocab_size = 3
        target_p = torch.tensor([0.5, 0.3, 0.2])
        # Convert to logits
        target_logits = torch.log(target_p + 1e-10)
        # Need gamma+1 logit positions (gamma=1 + bonus)
        logits = target_logits.unsqueeze(0).repeat(2, 1)

        sampler = SpeculativeSampler(mode='sampling', lenience=1.0)

        counts = [0, 0, 0]
        n_trials = 10000

        for _ in range(n_trials):
            # Draft a random token with uniform q
            draft_token = random.randint(0, vocab_size - 1)
            q = {draft_token: 1.0}

            accepted, n_acc, mask = sampler.verify(
                [draft_token], [q], logits, vocab_size=vocab_size
            )
            # The first accepted token should follow target distribution
            t = accepted[0]
            if 0 <= t < vocab_size:
                counts[t] += 1

        total = sum(counts)
        assert total > 0

        # Chi-square test
        expected = [target_p[i].item() * total for i in range(vocab_size)]
        chi2 = sum((counts[i] - expected[i]) ** 2 / max(expected[i], 1)
                   for i in range(vocab_size))

        # For df=2 (vocab_size-1), chi2 critical at p=0.01 is 9.21
        # We want chi2 < 9.21 (p > 0.01)
        assert chi2 < 9.21, (
            "Chi-square test failed: chi2=%.2f > 9.21 (p < 0.01). "
            "Counts=%s, expected=%s" % (chi2, counts, expected)
        )

    def test_lenience_monotonic_acceptance(self):
        """G3.1: Acceptance rate monotonically non-decreasing as l decreases."""
        random.seed(42)
        torch.manual_seed(42)

        vocab_size = 5
        target_p = torch.tensor([0.4, 0.25, 0.2, 0.1, 0.05])
        target_logits = torch.log(target_p + 1e-10)
        logits = target_logits.unsqueeze(0).repeat(2, 1)

        lenience_values = [1.0, 0.8, 0.6, 0.45, 0.3]
        acceptance_rates = []

        for l in lenience_values:
            sampler = SpeculativeSampler(mode='sampling', lenience=l)
            accepted_count = 0
            n_trials = 2000

            for _ in range(n_trials):
                draft_token = random.randint(0, vocab_size - 1)
                q = {draft_token: 1.0}
                _, n_acc, _ = sampler.verify(
                    [draft_token], [q], logits, vocab_size=vocab_size
                )
                accepted_count += n_acc

            rate = accepted_count / n_trials
            acceptance_rates.append(rate)

        # Acceptance should be non-decreasing as lenience decreases
        for i in range(len(acceptance_rates) - 1):
            assert acceptance_rates[i] <= acceptance_rates[i + 1] + 0.05, (
                "Acceptance not monotonic: l=%s rate=%.3f > l=%s rate=%.3f" % (
                    lenience_values[i], acceptance_rates[i],
                    lenience_values[i + 1], acceptance_rates[i + 1])
            )


# =========================================================================== #
#  Gamma controller tests (P3-4)
# =========================================================================== #

class TestGammaController(object):
    """Test the adaptive gamma controller."""

    def test_warmup_returns_moderate_gamma(self):
        ctrl = GammaController(gamma_max=6)
        # During warmup (first 5 steps), should return moderate gamma
        g = ctrl.get_gamma()
        assert 1 <= g <= 6

    def test_update_changes_alpha(self):
        ctrl = GammaController(gamma_max=4, ema_alpha=0.5)
        initial = ctrl.get_alpha_hat()[0]

        # All accepted
        ctrl.update([True, True, True], 3)
        after_accept = ctrl.get_alpha_hat()[0]
        assert after_accept > initial

        # All rejected
        ctrl.update([False, False, False], 3)
        after_reject = ctrl.get_alpha_hat()[0]
        assert after_reject < after_accept

    def test_v_table_affects_gamma(self):
        """With expensive V(k), gamma should be lower."""
        # Cheap verification
        v_cheap = {k: 0.01 * k for k in range(1, 9)}
        ctrl_cheap = GammaController(gamma_max=6, v_table=v_cheap)
        for _ in range(10):
            ctrl_cheap.update([True, True, True, True], 4)

        # Expensive verification
        v_expensive = {k: 0.5 * k for k in range(1, 9)}
        ctrl_exp = GammaController(gamma_max=6, v_table=v_expensive)
        for _ in range(10):
            ctrl_exp.update([True, True, True, True], 4)

        g_cheap = ctrl_cheap.get_gamma()
        g_exp = ctrl_exp.get_gamma()
        # With expensive verification, gamma should be <= cheap
        assert g_exp <= g_cheap

    def test_expected_tokens_formula(self):
        """E[tokens] with all alpha=1 should be gamma+1."""
        ctrl = GammaController(gamma_max=6)
        # Set all alphas to 1.0
        ctrl._alpha_hat = [1.0] * 7
        e = ctrl._expected_tokens(4)
        assert abs(e - 5.0) < 1e-6  # 1 + 4 = 5

    def test_expected_tokens_with_zero_alpha(self):
        """E[tokens] with alpha=0 should be 1."""
        ctrl = GammaController(gamma_max=6)
        ctrl._alpha_hat = [0.0] * 7
        e = ctrl._expected_tokens(4)
        assert abs(e - 1.0) < 1e-6

    def test_get_stats(self):
        ctrl = GammaController(gamma_max=4)
        ctrl.update([True, True], 2)
        stats = ctrl.get_stats()
        assert 'gamma_max' in stats
        assert 'alpha_hat' in stats
        assert stats['steps'] == 1


# =========================================================================== #
#  Integration test (lightweight — no actual model weights)
# =========================================================================== #

class TestDrafterSamplerIntegration(object):
    """Test drafter + sampler interaction without a real model."""

    def test_draft_and_verify_greedy(self):
        """Draft tokens, then verify with known logits."""
        drafter = SuffixDrafter(max_n=4, selector='importance')
        sampler = SpeculativeSampler(mode='greedy', lenience=1.0)

        # Build history with a repeating pattern
        pattern = [10, 20, 30, 40] * 5  # 20 tokens
        drafter.commit_tokens(pattern)

        # Draft
        draft_ids, draft_probs, match_len = drafter.draft(3)
        assert len(draft_ids) > 0

        # Create fake target logits that agree with the draft
        gamma = len(draft_ids)
        logits = torch.zeros(gamma + 1, 100)
        for i, tid in enumerate(draft_ids):
            logits[i, tid] = 10.0  # make draft token the argmax
        logits[gamma, 50] = 10.0  # bonus

        accepted, n_acc, mask = sampler.verify(draft_ids, draft_probs, logits)
        assert n_acc == gamma  # all should be accepted
        assert all(mask)

    def test_draft_and_verify_with_rejection(self):
        """Draft tokens, verify with logits that disagree."""
        drafter = SuffixDrafter(max_n=4, selector='importance')
        sampler = SpeculativeSampler(mode='greedy', lenience=1.0)

        pattern = [10, 20, 30, 40] * 5
        drafter.commit_tokens(pattern)

        draft_ids, draft_probs, _ = drafter.draft(3)
        if not draft_ids:
            return  # skip if no match

        gamma = len(draft_ids)
        logits = torch.zeros(gamma + 1, 100)
        # Make target disagree with all drafts
        for i in range(gamma):
            logits[i, 99] = 10.0  # always prefer token 99
        logits[gamma, 50] = 10.0

        accepted, n_acc, mask = sampler.verify(draft_ids, draft_probs, logits)
        assert n_acc == 0  # none accepted
        assert accepted[0] == 99  # correction to target's argmax


# =========================================================================== #
#  Demo validator tests (P3-6)
# =========================================================================== #

class TestDemoValidator(object):
    """Test WAF regex validator and triage evaluator."""

    def test_validate_waf_regexes_good(self):
        from tools.demo_akamai import validate_waf_regexes
        output = "Regex: ('\\ OR\\ )\nRegex: (<script[^>]*>)\n"
        results, rate = validate_waf_regexes(output)
        assert len(results) == 2
        assert rate == 100.0
        assert all(r['compiled'] for r in results)

    def test_validate_waf_regexes_bad(self):
        from tools.demo_akamai import validate_waf_regexes
        output = "Regex: ([invalid\n"
        results, rate = validate_waf_regexes(output)
        assert len(results) == 1
        assert rate == 0.0
        assert not results[0]['compiled']

    def test_evaluate_waf_recall(self):
        from tools.demo_akamai import evaluate_waf_recall
        regexes = [r"OR\s+'1'"]
        attack_lines = ["GET /search?q=' OR '1'='1 HTTP/1.1"]
        benign_lines = ["GET /index.html HTTP/1.1"]

        result = evaluate_waf_recall(regexes, attack_lines, benign_lines)
        assert result['recall'] == 100.0
        assert result['fpr'] == 0.0

    def test_evaluate_triage(self):
        from tools.demo_akamai import evaluate_triage
        labels = {
            'labels': [
                {'client_ip': '10.0.0.1'},
                {'client_ip': '10.0.0.1'},
                {'client_ip': '10.0.0.2'},
            ]
        }
        output = "Top offender: 10.0.0.1 with 2 hits"
        result = evaluate_triage(output, labels)
        assert result['top_offender_found'] is True
        assert result['top_offender_ip'] == '10.0.0.1'


# =========================================================================== #
#  G3.2: Speculative Greedy Equivalence Test
# =========================================================================== #

class TestSpeculativeGreedyEquivalence(object):
    """G3.2: With eviction off (budget >= context), speculative greedy at lenience 1.0
    must equal plain greedy for 20 prompts x 128 tokens using real OPT-125M weights.
    Allow <= 1% mismatches, and each mismatch must be a near-tie (top-2 logit margin < 0.05).
    """

    def test_greedy_equivalence_real_weights(self):
        from edgespec.model.loader import load_model
        from edgespec.spec.drafter import SuffixDrafter
        from edgespec.spec.sampler import SpeculativeSampler
        from edgespec.spec.verifier import SpeculativeVerifier

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(base_dir, 'data', 'weights', 'opt-125m-4bit.edgespec'),
            os.path.expanduser('~/edgespec/edgespec-h2o/data/weights/opt-125m-4bit.edgespec'),
            os.path.expanduser('~/edgespec/models/opt-125m'),
        ]
        weights_path = None
        for p in candidates:
            if os.path.exists(p):
                weights_path = p
                break

        if weights_path is None:
            pytest.skip("OPT-125M real weights not found for greedy equivalence test")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model(weights_path, device=device)
        model.eval()

        torch.manual_seed(42)
        random.seed(42)
        vocab_size = model.config['vocab_size']
        n_prompts = 20
        max_new_tokens = 128

        total_tokens = 0
        mismatches = 0
        near_ties = 0

        for pi in range(n_prompts):
            p_len = 16 + (pi % 17)
            prompt_ids = torch.randint(10, min(1000, vocab_size), (1, p_len), device=device, dtype=torch.long)

            # Plain greedy baseline
            model.reset_caches()
            model.init_kv_caches(device=device)
            baseline_full = model.generate_greedy(prompt_ids, max_new_tokens=max_new_tokens)
            baseline_gen = baseline_full[p_len:]

            # Speculative greedy with eviction off (budget >= 2048) and lenience 1.0
            model.reset_caches()
            model.init_h2o_caches(
                n_hh=2048,
                n_local=128,
                n_pin=4,
                n_scratch=8,
                decay=1.0,
                device=device
            )
            drafter = SuffixDrafter(max_n=4, selector='importance')
            sampler = SpeculativeSampler(mode='greedy', lenience=1.0)
            verifier = SpeculativeVerifier(
                model=model,
                drafter=drafter,
                sampler=sampler,
                gamma_fixed=3
            )
            spec_full, _ = verifier.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens
            )
            spec_gen = spec_full[p_len:]

            comp_len = min(len(baseline_gen), len(spec_gen))
            for t in range(comp_len):
                total_tokens += 1
                b_tok = baseline_gen[t]
                s_tok = spec_gen[t]
                if b_tok != s_tok:
                    mismatches += 1
                    # Verify near-tie: top-2 logit margin < 0.05
                    prefix = baseline_full[:p_len + t]
                    prefix_tensor = torch.tensor([prefix], device=device, dtype=torch.long)
                    model.reset_caches()
                    model.init_kv_caches(device=device)
                    with torch.no_grad():
                        logits = model.prefill_chunked(prefix_tensor)
                    last_logits = logits[0, -1, :]
                    top2 = torch.topk(last_logits, 2).values
                    margin = (top2[0] - top2[1]).item()
                    assert margin < 0.05, (
                        "Mismatch at prompt %d, token %d (b=%d, s=%d) is not a near-tie (margin=%.4f >= 0.05)"
                        % (pi, t, b_tok, s_tok, margin)
                    )
                    near_ties += 1
                    break

        mismatch_rate = (mismatches / float(total_tokens)) if total_tokens > 0 else 0.0
        assert mismatch_rate <= 0.01, (
            "Mismatch rate %.3f%% > 1.0%% (mismatches=%d, near_ties=%d, total=%d)"
            % (mismatch_rate * 100.0, mismatches, near_ties, total_tokens)
        )
