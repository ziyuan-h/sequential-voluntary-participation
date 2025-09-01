import itertools
from typing import Sequence, Optional, Tuple

import numpy as np


class LinearQuadraticGame:

    def __init__(self, alpha, phi, g) -> None:
        self.alpha = alpha
        self.phi = phi
        self.g = g

    @property
    def num_agents(self) -> int:
        return self.g.shape[0]

    def compute_utility(self, x: np.ndarray) -> np.ndarray:
        return self.alpha * x - (1/2) * x**2 + self.phi * x * (self.g @ x)

    def compute_nash_equilibrium(self) -> np.ndarray:
        N = self.g.shape[0]
        return np.linalg.inv(np.eye(N) - self.phi * self.g) @ self.alpha

    def compute_social_optimum(self) -> np.ndarray:
        N = self.g.shape[0]
        return np.linalg.inv(np.eye(N) - self.phi * (self.g + self.g.T)) @ self.alpha

    def compute_incentive_compatible_profile(self, outliers: Sequence[int]) -> np.ndarray:
        N = self.g.shape[0]
        participants = [i for i in range(N) if i not in outliers]
        g_copy = self.g.copy()
        if participants:
            g_copy[participants, :] = (self.g + self.g.T)[participants, :]

        return np.linalg.inv(np.eye(N) - self.phi * g_copy) @ self.alpha

    def equilibrium_tax(self, mechanism: str = "pivotal",
                        ordering: Optional[Sequence[int]] = None) -> np.ndarray:
        """
        The returned value is the pivotal or sequential tax exerted on each individual.
        The returned taxes are ordered in [0,1,2,...,N-1].
        "ordering" specifies the "taxation order" rather than the returned order.
        This is a positive value as opposed to the negative "subsidy" used in the paper.
        """
        N = self.g.shape[0]
        x_social = self.compute_social_optimum()
        u_social = self.compute_utility(x_social)
        u_ic = []
        ordering = ordering if ordering is not None else list(range(N))
        for i, agent in enumerate(ordering):
            if mechanism == "pivotal":
                x_ic = self.compute_incentive_compatible_profile([agent])
            elif mechanism == "sequential":
                x_ic = self.compute_incentive_compatible_profile(ordering[i:])
            else:
                raise ValueError("Unknown mechanism.")

            u_ic.append(self.compute_utility(x_ic)[i])

        u_ic = np.asarray(u_ic)
        return u_social - u_ic

    def find_optimal_ranking_exhaustive(self) -> Tuple[float, Sequence]:
        max_tax, max_tax_perm = -np.inf, None
        for perm in itertools.permutations(range(self.num_agents)):
            total_tax = self.equilibrium_tax(mechanism="sequential", ordering=perm).sum()
            if total_tax > max_tax:
                max_tax, max_tax_perm = total_tax, perm

        return max_tax, max_tax_perm