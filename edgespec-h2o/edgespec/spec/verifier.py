# -*- coding: utf-8 -*-
"""Verifier loop — ties the drafter, model, cache, and sampler together (P3-3).

Implements the full speculative decoding verify cycle:
1. Drafter produces gamma draft tokens.
2. forward_block_h2o over [last_committed] + gamma drafts.
3. Speculative sampler accepts/rejects.
4. commit(n_accepted + 1), update_scores, evict_to_budget.

Python-3.6 safe.
"""

import time

import torch


class SpeculativeVerifier(object):
    """Full speculative decoding engine.

    Parameters
    ----------
    model : OPTModel
        The target model (with H2O caches initialised).
    drafter : SuffixDrafter
        The zero-RAM drafter.
    sampler : SpeculativeSampler
        The acceptance sampler.
    gamma_ctrl : GammaController or None
        Adaptive gamma controller.  If None, uses fixed gamma.
    gamma_fixed : int
        Fixed gamma if no controller is provided.
    importance_layers : list of int
        Layer indices to accumulate importance from for the drafter.
    """

    def __init__(self, model, drafter, sampler, gamma_ctrl=None,
                 gamma_fixed=4, importance_layers=None):
        self.model = model
        self.drafter = drafter
        self.sampler = sampler
        self.gamma_ctrl = gamma_ctrl
        self.gamma_fixed = gamma_fixed
        self.importance_layers = importance_layers or []

        # Telemetry accumulators
        self.stats = {
            'total_tokens': 0,
            'total_steps': 0,
            'total_draft_tokens': 0,
            'total_accepted': 0,
            'total_rejected': 0,
            'total_bonus': 0,
            'step_times': [],
            'acceptance_history': [],
            'gamma_history': [],
            'match_lengths': [],
        }

    def reset_stats(self):
        """Clear telemetry."""
        self.stats = {
            'total_tokens': 0,
            'total_steps': 0,
            'total_draft_tokens': 0,
            'total_accepted': 0,
            'total_rejected': 0,
            'total_bonus': 0,
            'step_times': [],
            'acceptance_history': [],
            'gamma_history': [],
            'match_lengths': [],
        }

    def _get_gamma(self):
        """Get the gamma value for this step."""
        if self.gamma_ctrl is not None:
            return self.gamma_ctrl.get_gamma()
        return self.gamma_fixed

    def _extract_importance(self, all_attn_probs, n_keep):
        """Extract token importance from attention probabilities.

        Uses the configured importance_layers to accumulate a per-position
        importance score from the attention probabilities.

        Parameters
        ----------
        all_attn_probs : list of [n_heads, n_new, n_total]
        n_keep : int

        Returns
        -------
        list of float, one per committed token
        """
        if not self.importance_layers or all_attn_probs is None:
            return [1.0] * n_keep

        # Average importance across configured layers
        importance = None
        n_layers_used = 0
        for layer_idx in self.importance_layers:
            if layer_idx < len(all_attn_probs) and all_attn_probs[layer_idx] is not None:
                probs = all_attn_probs[layer_idx]  # [n_heads, n_new, n_total]
                # Sum across heads and query positions -> [n_total]
                layer_imp = probs.float().sum(dim=0).sum(dim=0)
                if importance is None:
                    importance = layer_imp
                else:
                    # Match sizes
                    min_len = min(importance.shape[0], layer_imp.shape[0])
                    importance = importance[:min_len] + layer_imp[:min_len]
                n_layers_used += 1

        if importance is None:
            return [1.0] * n_keep

        # The committed tokens are the last n_keep entries
        scores = []
        for i in range(n_keep):
            idx = importance.shape[0] - n_keep + i
            if 0 <= idx < importance.shape[0]:
                scores.append(importance[idx].item())
            else:
                scores.append(1.0)
        return scores

    @torch.no_grad()
    def step(self):
        """Execute one speculative decoding step.

        Returns
        -------
        accepted_tokens : list of int
            Token IDs produced this step.
        step_info : dict
            Telemetry for this step.
        """
        t_start = time.time()
        device = next(self.model.parameters()).device

        gamma = self._get_gamma()

        # --- Draft ---
        draft_ids, draft_probs, match_len = self.drafter.draft(gamma)
        actual_gamma = len(draft_ids)

        if actual_gamma == 0:
            # No match -> normal 1-token step
            return self._single_token_step(t_start)

        # --- Verify ---
        # Build input: [last_committed_token] + draft_ids
        last_token = self.drafter.history[-1]
        verify_input = [last_token] + draft_ids
        input_ids = torch.tensor([verify_input], device=device, dtype=torch.long)

        n_verify = len(verify_input)  # actual_gamma + 1

        # Begin block on all H2O caches
        for cache in self.model.h2o_caches:
            cache.begin_block(n_verify)

        # Forward pass
        logits, all_attn_probs = self.model.forward_block_h2o(
            input_ids, return_attn_probs=True
        )
        # logits: [1, n_verify, vocab_size]
        # logits[0, 0] -> logits for verifying draft_ids[0]
        # logits[0, i] -> logits for verifying draft_ids[i]
        # logits[0, actual_gamma] -> bonus position

        # Extract verification logits
        verify_logits = logits[0]  # [n_verify, vocab_size]

        # --- Accept/Reject ---
        accepted_ids, n_accepted, acceptance_mask = self.sampler.verify(
            draft_ids, draft_probs, verify_logits,
            vocab_size=logits.shape[-1]
        )

        # Number of tokens to commit (n_accepted drafts + 1 correction/bonus)
        n_commit = len(accepted_ids)

        # --- Commit ---
        # We need to commit n_commit tokens in the cache
        # The first token in the verify input was the re-processed last token,
        # so we commit positions 1..n_commit (the draft/correction tokens)
        # But the cache already has the last token, so we committed
        # n_verify new KV entries; keep n_commit of them (skip the first
        # which is the already-committed last token... but actually the
        # cache's begin_block reserved slots for ALL n_verify tokens).
        #
        # Important nuance: the first position in our verify block is the
        # re-run of the last committed token. In the H2O cache, we've
        # already committed that token. But forward_block_h2o appended
        # KV for it again. We need to handle this carefully.
        #
        # Actually, the verifier forward produces KV for n_verify positions.
        # We want to keep KV for positions 1..n_commit (the new tokens).
        # Position 0's KV is redundant (it's the last committed token).
        # But since we can only commit contiguous from the start, we need
        # to commit all n_commit entries and the model handles it.
        #
        # Simpler approach: commit n_commit, which includes the re-run of
        # last token + accepted tokens. The cache handles dedup.
        for i, cache in enumerate(self.model.h2o_caches):
            cache.commit(n_commit)
            if all_attn_probs is not None and all_attn_probs[i] is not None:
                cache.update_scores(all_attn_probs[i][:, :n_commit, :], n_commit)
            cache.evict_to_budget()

        # Extract importance for committed tokens
        importance_scores = self._extract_importance(all_attn_probs, n_commit)

        # Update drafter with accepted tokens (skip the first which is re-processed)
        # accepted_ids already contains only the new tokens
        self.drafter.commit_tokens(accepted_ids, importance_scores[:len(accepted_ids)])

        # --- Telemetry ---
        t_end = time.time()
        step_time = t_end - t_start

        is_bonus = (n_accepted == actual_gamma)
        self.stats['total_tokens'] += len(accepted_ids)
        self.stats['total_steps'] += 1
        self.stats['total_draft_tokens'] += actual_gamma
        self.stats['total_accepted'] += n_accepted
        self.stats['total_rejected'] += (actual_gamma - n_accepted)
        if is_bonus:
            self.stats['total_bonus'] += 1
        self.stats['step_times'].append(step_time)
        self.stats['acceptance_history'].append(n_accepted)
        self.stats['gamma_history'].append(actual_gamma)
        self.stats['match_lengths'].append(match_len)

        # Update gamma controller
        if self.gamma_ctrl is not None:
            self.gamma_ctrl.update(acceptance_mask, actual_gamma)

        step_info = {
            'gamma': actual_gamma,
            'n_accepted': n_accepted,
            'n_produced': len(accepted_ids),
            'is_bonus': is_bonus,
            'match_length': match_len,
            'step_time_ms': step_time * 1000,
            'tok_per_sec': len(accepted_ids) / step_time if step_time > 0 else 0,
            'acceptance_mask': acceptance_mask,
        }

        return accepted_ids, step_info

    def _single_token_step(self, t_start):
        """Fallback: generate one token without speculation."""
        device = next(self.model.parameters()).device
        last_token = self.drafter.history[-1]
        input_ids = torch.tensor([[last_token]], device=device, dtype=torch.long)

        # Begin block for 1 token
        for cache in self.model.h2o_caches:
            cache.begin_block(1)

        logits, all_attn_probs = self.model.forward_block_h2o(
            input_ids, return_attn_probs=True
        )

        if self.sampler.mode == 'greedy':
            next_token = logits[0, -1, :].argmax(dim=-1).item()
        else:
            p = torch.nn.functional.softmax(
                logits[0, -1, :].float() / self.sampler.temperature, dim=-1
            )
            next_token = torch.multinomial(p, 1).item()

        # Commit
        for i, cache in enumerate(self.model.h2o_caches):
            cache.commit(1)
            if all_attn_probs is not None and all_attn_probs[i] is not None:
                cache.update_scores(all_attn_probs[i], 1)
            cache.evict_to_budget()

        # Extract importance and update drafter
        importance_scores = self._extract_importance(all_attn_probs, 1)
        self.drafter.commit_tokens([next_token], importance_scores)

        t_end = time.time()
        step_time = t_end - t_start

        self.stats['total_tokens'] += 1
        self.stats['total_steps'] += 1
        self.stats['step_times'].append(step_time)
        self.stats['gamma_history'].append(0)
        self.stats['acceptance_history'].append(0)
        self.stats['match_lengths'].append(0)

        if self.gamma_ctrl is not None:
            self.gamma_ctrl.update([], 0)

        step_info = {
            'gamma': 0,
            'n_accepted': 0,
            'n_produced': 1,
            'is_bonus': False,
            'match_length': 0,
            'step_time_ms': step_time * 1000,
            'tok_per_sec': 1.0 / step_time if step_time > 0 else 0,
            'acceptance_mask': [],
        }

        return [next_token], step_info

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=128, chunk_size=128,
                 callback=None):
        """Full speculative generation loop.

        Parameters
        ----------
        prompt_ids : torch.LongTensor [1, prompt_len]
        max_new_tokens : int
        chunk_size : int
        callback : callable or None
            Called after each step with (accepted_tokens, step_info, all_ids).

        Returns
        -------
        all_ids : list of int
            All token IDs (prompt + generated).
        gen_stats : dict
            Aggregated generation statistics.
        """
        device = prompt_ids.device

        # Reset drafter
        self.drafter.reset()
        self.reset_stats()

        # Prefill with H2O
        logits = self.model.prefill_h2o(prompt_ids, chunk_size=chunk_size)

        # First token (from prefill)
        if self.sampler.mode == 'greedy':
            first_token = logits[0, -1, :].argmax(dim=-1).item()
        else:
            p = torch.nn.functional.softmax(
                logits[0, -1, :].float() / self.sampler.temperature, dim=-1
            )
            first_token = torch.multinomial(p, 1).item()

        # Initialise drafter history with prompt + first token
        prompt_list = prompt_ids[0].tolist()
        self.drafter.commit_tokens(prompt_list + [first_token])

        all_ids = prompt_list + [first_token]
        eos_id = self.model.config.get('eos_token_id', 2)
        n_generated = 1

        while n_generated < max_new_tokens:
            if all_ids[-1] == eos_id:
                break

            accepted_tokens, step_info = self.step()

            for tid in accepted_tokens:
                all_ids.append(tid)
                n_generated += 1
                if n_generated >= max_new_tokens:
                    break
                if tid == eos_id:
                    break

            if callback is not None:
                callback(accepted_tokens, step_info, all_ids)

        # Aggregated stats
        total_time = sum(self.stats['step_times'])
        gen_stats = {
            'total_tokens': self.stats['total_tokens'],
            'total_steps': self.stats['total_steps'],
            'total_draft_tokens': self.stats['total_draft_tokens'],
            'total_accepted': self.stats['total_accepted'],
            'total_rejected': self.stats['total_rejected'],
            'total_bonus': self.stats['total_bonus'],
            'mean_accepted_per_step': (
                self.stats['total_accepted'] / max(1, self.stats['total_steps'])
            ),
            'alpha_hat': (
                self.stats['total_accepted'] /
                max(1, self.stats['total_draft_tokens'])
            ),
            'total_time_s': total_time,
            'tokens_per_sec': (
                self.stats['total_tokens'] / total_time if total_time > 0 else 0
            ),
            'mean_gamma': (
                sum(self.stats['gamma_history']) /
                max(1, len(self.stats['gamma_history']))
            ),
            'n_generated': n_generated,
        }

        return all_ids, gen_stats

    def get_telemetry(self):
        """Return current telemetry snapshot for the UI.

        Returns
        -------
        dict
        """
        total_time = sum(self.stats['step_times'])
        n_steps = max(1, self.stats['total_steps'])

        # Per-position acceptance rates (EMA) for the last 100 steps
        recent = self.stats['acceptance_history'][-100:]
        recent_gammas = self.stats['gamma_history'][-100:]

        return {
            'total_tokens': self.stats['total_tokens'],
            'total_steps': self.stats['total_steps'],
            'tokens_per_sec': (
                self.stats['total_tokens'] / total_time if total_time > 0 else 0
            ),
            'alpha_hat': (
                self.stats['total_accepted'] /
                max(1, self.stats['total_draft_tokens'])
            ),
            'mean_accepted_per_step': self.stats['total_accepted'] / n_steps,
            'mean_gamma': sum(recent_gammas) / max(1, len(recent_gammas)),
            'recent_accepted': recent[-20:],
            'recent_gammas': recent_gammas[-20:],
        }
