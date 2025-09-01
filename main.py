import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from util import generate_random_symmetric_binary_matrix, draw_network, get_one_divide_spectral_radius
from linear_quadratic import LinearQuadraticGame


def run():
    phi = 0.05
    result_history = []
    for n in range(3, 9):
        alpha = np.ones(n)
        g_list = []
        while len(g_list) < 3:
            random_g = generate_random_symmetric_binary_matrix(n)
            if get_one_divide_spectral_radius(random_g + random_g.T) > phi:
                g_list.append(random_g)

        record = []
        for g in tqdm(g_list, desc=f"n={n}"):
            game = LinearQuadraticGame(alpha=alpha, phi=phi, g=g)
            max_tax, max_tax_perm = game.find_optimal_ranking_exhaustive()
            pivotal_tax_sequence = game.equilibrium_tax(mechanism="pivotal")
            pivotal_ordering = np.argsort(pivotal_tax_sequence)
            seq_tax_by_pivotal_order = game.equilibrium_tax(mechanism="sequential", ordering=pivotal_ordering)
            seq_tax_total = np.sum(seq_tax_by_pivotal_order)
            record.append({"optimal": max_tax, "pivotal": seq_tax_total, "network": g})

        result_history.append(record)

    opts = [np.mean([item['optimal'] for item in exp]) for exp in result_history]
    pivot = [np.mean([item['pivotal'] for item in exp]) for exp in result_history]
    plt.plot(np.vstack([opts, pivot]).T)
    plt.legend(["Opt", "Seq"])
    plt.show()

    return result_history


if __name__ == "__main__":
    run()