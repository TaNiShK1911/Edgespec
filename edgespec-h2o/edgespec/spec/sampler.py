# -*- coding: utf-8 -*-
"""Speculative sampler — acceptance/rejection logic (P3-2).

Implements the speculative decoding acceptance rule from Leviathan et al.
(arXiv 2211.17192), Algorithm 1 with lenience extension (Appendix A.5).

Two modes:
- **Greedy**: accept iff x_hat == argmax(p), or with lenience: p(x_hat) >= l * max(p).
- **Sampling**: accept if l*q(x_hat) <= p(x_hat), else accept with prob p/(l*q).
  On reject, resample from norm(max(0, p - l*q)).

l = 1 is the strict/lossless setting.  l < 1 trades fidelity for acceptance.

Python-3.6 safe.
"""

import math
import random

import torch
import torch.nn.functional as F


class SpeculativeSampler(object):
    """Speculative decoding acceptance sampler.

    Parameters
    ----------
    mode : str
        'greedy' or 'sampling'.
    lenience : float
        Lenience parameter l (1.0 = strict).
    temperature : float
        Sampling temperature (only used in sampling mode, default 1.0).
    """

    def __init__(self, mode='greedy', lenience=1.0, temperature=1.0):
        assert mode in ('greedy', 'sampling'), \
            "mode must be 'greedy' or 'sampling', got '%s'" % mode
        assert 0.0 < lenience <= 1.0, \
            "lenience must be in (0, 1], got %s" % lenience
        self.mode = mode
        self.lenience = lenience
        self.temperature = temperature

    def verify(self, draft_ids, draft_probs, target_logits, vocab_size=None):
        """Run the speculative acceptance/rejection loop.

        Parameters
        ----------
        draft_ids : list of int
            Draft token IDs, length gamma.
        draft_probs : list of dict
            Draft distribution q for each position.
            Each dict maps {token_id: float}.  For greedy drafts, this is
            typically one-hot {draft_id: 1.0}.
        target_logits : torch.Tensor [gamma + 1, vocab_size]
            Target model logits for positions [last_committed, draft_0, ..., draft_{gamma-1}].
            So target_logits[0] gives logits for verifying draft_ids[0],
            target_logits[i] gives logits for verifying draft_ids[i],
            target_logits[gamma] is the bonus position.
        vocab_size : int or None
            If None, inferred from target_logits.

        Returns
        -------
        accepted_ids : list of int
            Accepted token IDs (1 to gamma+1 tokens).
        n_accepted : int
            Number of draft tokens accepted (0 to gamma).
        acceptance_mask : list of bool
            Per-draft acceptance decisions.
        """
        gamma = len(draft_ids)
        if vocab_size is None:
            vocab_size = target_logits.shape[-1]

        if self.mode == 'greedy':
            return self._verify_greedy(draft_ids, draft_probs,
                                       target_logits, vocab_size)
        else:
            return self._verify_sampling(draft_ids, draft_probs,
                                         target_logits, vocab_size)

    def _verify_greedy(self, draft_ids, draft_probs, target_logits, vocab_size):
        """Greedy verification: accept iff draft == argmax(target).

        With lenience (l < 1): accept if p(x_hat) >= l * max(p).
        """
        gamma = len(draft_ids)
        accepted_ids = []
        acceptance_mask = []
        n_accepted = 0

        for i in range(gamma):
            # Target distribution for position i
            logits_i = target_logits[i]
            if self.temperature != 1.0:
                logits_i = logits_i / self.temperature
            p = F.softmax(logits_i.float(), dim=-1)

            draft_token = draft_ids[i]
            p_draft = p[draft_token].item()
            p_max = p.max().item()
            argmax_token = p.argmax().item()

            if self.lenience >= 1.0:
                # Strict: accept iff draft == argmax
                accept = (draft_token == argmax_token)
            else:
                # Lenience: accept if p(x_hat) >= l * max(p)
                accept = (p_draft >= self.lenience * p_max)

            acceptance_mask.append(accept)

            if accept:
                accepted_ids.append(draft_token)
                n_accepted += 1
            else:
                # Reject: use argmax of target as the correction token
                accepted_ids.append(argmax_token)
                break

        # If all gamma drafts accepted, sample a bonus token from p_{gamma}
        if n_accepted == gamma:
            bonus_logits = target_logits[gamma]
            if self.temperature != 1.0:
                bonus_logits = bonus_logits / self.temperature
            bonus_token = bonus_logits.argmax().item()
            accepted_ids.append(bonus_token)

        return accepted_ids, n_accepted, acceptance_mask

    def _verify_sampling(self, draft_ids, draft_probs, target_logits, vocab_size):
        """Sampling verification per Algorithm 1 + Appendix A.5.

        For draft x_hat with draft prob q and target prob p:
        - Accept if l*q(x_hat) <= p(x_hat)
        - Else accept with probability p(x_hat) / (l * q(x_hat))
        - On reject: resample from norm(max(0, p - l*q))
        """
        gamma = len(draft_ids)
        accepted_ids = []
        acceptance_mask = []
        n_accepted = 0

        for i in range(gamma):
            logits_i = target_logits[i]
            if self.temperature != 1.0:
                logits_i = logits_i / self.temperature
            p = F.softmax(logits_i.float(), dim=-1)

            draft_token = draft_ids[i]
            q_dict = draft_probs[i]

            # Get q(x_hat)
            q_draft = q_dict.get(draft_token, 0.0)

            # Get p(x_hat)
            p_draft = p[draft_token].item()

            # Acceptance decision
            l_q = self.lenience * q_draft

            if l_q <= p_draft:
                # Deterministic accept
                accept = True
            elif l_q > 0:
                # Accept with probability p / (l*q)
                accept_prob = p_draft / l_q
                accept = (random.random() < accept_prob)
            else:
                accept = True  # q = 0 means we always accept

            acceptance_mask.append(accept)

            if accept:
                accepted_ids.append(draft_token)
                n_accepted += 1
            else:
                # Reject: resample from norm(max(0, p - l*q))
                correction_token = self._resample_correction(
                    p, q_dict, vocab_size
                )
                accepted_ids.append(correction_token)
                break

        # If all accepted: sample bonus token from p_{gamma}
        if n_accepted == gamma:
            bonus_logits = target_logits[gamma]
            if self.temperature != 1.0:
                bonus_logits = bonus_logits / self.temperature
            bonus_p = F.softmax(bonus_logits.float(), dim=-1)
            bonus_token = torch.multinomial(bonus_p, 1).item()
            accepted_ids.append(bonus_token)

        return accepted_ids, n_accepted, acceptance_mask

    def _resample_correction(self, p, q_dict, vocab_size):
        """Resample from norm(max(0, p - l*q)).

        Parameters
        ----------
        p : torch.Tensor [vocab_size]
            Target probabilities.
        q_dict : dict {int: float}
            Draft distribution.
        vocab_size : int

        Returns
        -------
        int : resampled token ID
        """
        # Build q tensor
        q = torch.zeros(vocab_size, dtype=torch.float32, device=p.device)
        for tid, prob in q_dict.items():
            if tid < vocab_size:
                q[tid] = prob

        # Correction distribution: max(0, p - l*q)
        correction = torch.clamp(p - self.lenience * q, min=0.0)
        total = correction.sum()
        if total <= 0:
            # Fallback: sample from p
            return torch.multinomial(p, 1).item()

        correction = correction / total
        return torch.multinomial(correction, 1).item()
