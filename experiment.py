import random
from typing import Dict, List

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from util import generate_random_symmetric_binary_matrix, draw_network, get_one_divide_spectral_radius
from linear_quadratic import LinearQuadraticGame


def get_pivotal_ordering(game: LinearQuadraticGame) -> Dict[str, np.ndarray]:
    pivotal_tax_sequence = game.equilibrium_tax(mechanism="pivotal")
    pivotal_ordering = np.argsort(pivotal_tax_sequence)[::-1]
    return {"test_ordering": pivotal_ordering, "pivotal_tax": pivotal_tax_sequence}

def get_greedy_ordering(game: LinearQuadraticGame) -> Dict[str, np.ndarray]:
    ordering = []
    order_values = []
    # x_social = game.compute_social_optimum()
    # u_social = game.compute_utility(x_social)
    while len(ordering) < game.num_agents:
        outliers = list(set(range(game.num_agents)) - set(ordering))
        x_ic = game.compute_incentive_compatible_profile(outliers)
        # u_ic = game.compute_utility(x_ic)
        min_util, min_util_outlier = np.inf, None
        for agent in outliers:
            if x_ic[agent] < min_util:
                min_util, min_util_outlier = x_ic[agent], agent

        ordering.append(min_util_outlier)
        order_values.append(min_util)

    return {"test_ordering": np.asarray(ordering), "order_values": np.asarray(order_values)}

def get_random_ordering(game: LinearQuadraticGame) -> Dict[str, np.ndarray]:
    ordering = list(range(game.num_agents))
    random.shuffle(ordering)
    return {"test_ordering": np.asarray(ordering)}


def get_most_independent_ordering(game: LinearQuadraticGame) -> Dict[str, np.ndarray]:
    ordering = []
    order_values = []
    while len(ordering) < game.num_agents:
        outliers = set(range(game.num_agents)) - set(ordering)
        x_ic = game.compute_incentive_compatible_profile(list(outliers))
        u_ic = game.compute_utility(x_ic)
        min_diff, selected_outlier = np.inf, None
        for outlier in outliers:
            x_ic_with_outlier = game.compute_incentive_compatible_profile(list(outliers - {outlier}))
            u_ic_with_outlier = game.compute_incentive_compatible_profile(x_ic_with_outlier)
            u_diff = u_ic_with_outlier[outlier] - u_ic[outlier]
            if u_diff < min_diff:
                min_diff, selected_outlier = u_diff, outlier

        ordering.append(selected_outlier)
        order_values.append(min_diff)

    return {"test_ordering": np.asarray(ordering), "order_values": np.asarray(order_values)}


def compare_taxes(alpha: np.ndarray, phi: float, g: np.ndarray, test_type: str = "greedy") -> Dict[str, np.ndarray]:
    game = LinearQuadraticGame(alpha=alpha, phi=phi, g=g)
    max_tax, max_tax_perm = game.find_optimal_ranking_exhaustive()
    if test_type == "greedy":
        test_fields = get_greedy_ordering(game)
    elif test_type == "pivotal":
        test_fields = get_pivotal_ordering(game)
    elif test_type == "random":
        test_fields = get_random_ordering(game)
    elif test_type == "most_independent":
        test_fields = get_most_independent_ordering(game)
    else:
        raise ValueError("test_type must be 'greedy' or 'pivotal'")
    seq_tax_by_test_ordering = game.equilibrium_tax(mechanism="sequential", ordering=test_fields["test_ordering"])
    seq_tax_total = np.sum(seq_tax_by_test_ordering)
    return {"opt": max_tax, "optimal_rank": max_tax_perm,
            "test": seq_tax_total, "seq_tax_by_test": seq_tax_by_test_ordering,
            "network": g, **test_fields}

def plot_comparison(result_history: List[List[Dict[str, np.ndarray]]]) -> None:
    opts = [np.mean([item['opt'] for item in exp]) for exp in result_history]
    pivot = [np.mean([item['test'] for item in exp]) for exp in result_history]
    plt.plot(np.vstack([opts, pivot]).T)
    plt.legend(["Opt", "Seq"])
    plt.show()


def same_cost_diff_network():
    phi = 0.05
    result_history = []
    n_list = list(range(3, 9))
    num_trials = 10
    progress_bar = tqdm(total=len(n_list) * num_trials)
    for n in n_list:
        alpha = np.ones(n)
        g_list = []
        while len(g_list) < num_trials:
            random_g = generate_random_symmetric_binary_matrix(n)
            if get_one_divide_spectral_radius(random_g + random_g.T) > phi:
                g_list.append(random_g)
        record = []
        for g in g_list:
            record.append(compare_taxes(alpha, phi, g))
            progress_bar.update(1)
            progress_bar.set_description(f"N={n}")

        result_history.append(record)
    return result_history


def diff_cost_same_network():
    phi = 0.1
    result_history = []
    n_list = list(range(3, 9))
    num_trials = 10
    progress_bar = tqdm(total=len(n_list) * num_trials)
    for n in n_list:
        g = np.ones((n,n)) - np.eye(n)
        record = []
        for _ in range(num_trials):
            alpha = np.random.rand(n) * 2
            record.append(compare_taxes(alpha, phi, g))
            progress_bar.update(1)
            progress_bar.set_description(f"N={n}")

        result_history.append(record)
    return result_history