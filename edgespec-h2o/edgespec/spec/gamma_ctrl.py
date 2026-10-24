# -*- coding: utf-8 -*-
"""Adaptive gamma controller (P3-4).

Extends the paper's section 3.5 by using per-position acceptance EMA
and the *measured* V(k) table from Gate 1 (G1.9) instead of the paper's
flat-cost assumption.

Chooses gamma in {0, ..., gamma_max} maximizing:
    E[tokens](gamma) / (V(gamma + 1) + gamma * c)

where:
    E[tokens](gamma) = 1 + sum_{j=1..gamma} prod_{i<=j} alpha_hat_i

and alpha_hat_i is an EMA of the acceptance rate at draft position i.

Python-3.6 safe.
"""


class GammaController(object):
    """Adaptive gamma controller using measured V(k).

    Parameters
    ----------
    gamma_max : int
        Maximum gamma to consider (default 6).
    v_table : dict or None
        Measured V(k) table from G1.9.  Keys are int k, values are
        float time in seconds.  If None, uses a flat-cost model.
    drafter_cost : float
        Per-draft-token cost c (default 0.0 for zero-RAM drafter).
    ema_alpha : float
        EMA smoothing factor for acceptance rates (default 0.1).
    min_gamma : int
        Minimum gamma when a match is found (default 1).
    """

    def __init__(self, gamma_max=6, v_table=None, drafter_cost=0.0,
                 ema_alpha=0.1, min_gamma=1):
        self.gamma_max = gamma_max
        self.drafter_cost = drafter_cost
        self.ema_alpha = ema_alpha
        self.min_gamma = min_gamma

        # V(k) table: time to verify k tokens
        if v_table is not None:
            self._v = dict(v_table)
        else:
            self._v = None

        # Per-position acceptance EMA: alpha_hat[i] for position i (1-indexed)
        self._alpha_hat = [0.5] * (gamma_max + 1)  # initialise at 0.5

        # Per-match-length-bucket acceptance EMA
        self._alpha_by_match = {}

        # Step count for warmup
        self._steps = 0

    def get_gamma(self):
        """Compute the optimal gamma for this step.

        Returns
        -------
        int
        """
        if self._steps < 5:
            # Warmup: use a moderate fixed gamma
            return min(3, self.gamma_max)

        best_gamma = 0
        best_efficiency = 0.0

        for g in range(self.min_gamma, self.gamma_max + 1):
            e_tokens = self._expected_tokens(g)
            cost = self._verify_cost(g + 1) + g * self.drafter_cost
            if cost <= 0:
                cost = 1e-9
            efficiency = e_tokens / cost
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_gamma = g

        # If efficiency at best_gamma <= efficiency at gamma=0 (1 token step),
        # prefer gamma=0 only if there really is no benefit
        e0 = 1.0
        c0 = self._verify_cost(1)
        if c0 <= 0:
            c0 = 1e-9
        eff0 = e0 / c0

        if best_efficiency <= eff0:
            return 0

        return best_gamma

    def _expected_tokens(self, gamma):
        """Compute E[tokens] for a given gamma.

        E[tokens] = 1 + sum_{j=1..gamma} prod_{i=1..j} alpha_hat_i

        Without the i.i.d. assumption — uses per-position EMA.
        """
        e = 1.0
        prod = 1.0
        for j in range(1, gamma + 1):
            idx = min(j, len(self._alpha_hat)) - 1
            prod *= self._alpha_hat[idx]
            e += prod
        return e

    def _verify_cost(self, k):
        """Get V(k) — cost of verifying k tokens.

        Uses the measured table if available, otherwise linear extrapolation.
        """
        if self._v is not None:
            if k in self._v:
                return self._v[k]
            # Linear interpolation/extrapolation from available data
            keys = sorted(self._v.keys())
            if k < keys[0]:
                return self._v[keys[0]]
            if k > keys[-1]:
                # Extrapolate from last two points
                if len(keys) >= 2:
                    k1, k2 = keys[-2], keys[-1]
                    slope = (self._v[k2] - self._v[k1]) / (k2 - k1)
                    return self._v[k2] + slope * (k - k2)
                return self._v[keys[-1]]
            # Interpolate
            for i in range(len(keys) - 1):
                if keys[i] <= k <= keys[i + 1]:
                    t = float(k - keys[i]) / (keys[i + 1] - keys[i])
                    return self._v[keys[i]] * (1 - t) + self._v[keys[i + 1]] * t
        # Flat cost fallback
        return k * 0.01  # rough estimate

    def update(self, acceptance_mask, gamma):
        """Update EMAs after an acceptance decision.

        Parameters
        ----------
        acceptance_mask : list of bool
            Per-position acceptance decisions.
        gamma : int
            The gamma used for this step.
        """
        self._steps += 1

        for i, accepted in enumerate(acceptance_mask):
            idx = min(i, len(self._alpha_hat) - 1)
            value = 1.0 if accepted else 0.0
            self._alpha_hat[idx] = (
                (1 - self.ema_alpha) * self._alpha_hat[idx]
                + self.ema_alpha * value
            )

    def get_alpha_hat(self):
        """Return current per-position acceptance EMAs.

        Returns
        -------
        list of float
        """
        return list(self._alpha_hat)

    def set_v_table(self, v_table):
        """Update the V(k) table (e.g. from G1.9 results).

        Parameters
        ----------
        v_table : dict {int: float}
        """
        self._v = dict(v_table)

    def get_stats(self):
        """Return controller statistics for telemetry.

        Returns
        -------
        dict
        """
        return {
            'gamma_max': self.gamma_max,
            'steps': self._steps,
            'alpha_hat': list(self._alpha_hat),
            'current_gamma': self.get_gamma(),
        }
