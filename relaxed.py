import itertools

import numpy as np
from tqdm import trange


def create_adjacency_matrix(N):
    """Creates a random symmetric adjacency matrix (A) with zero diagonal."""
    np.random.seed(42)
    A = np.random.rand(N, N)
    A = (A + A.T) / 2  # Make it symmetric
    np.fill_diagonal(A, 0)
    A = A / (N * np.max(A))
    return A


def create_mask_matrix(N, i):
    """Creates the mask matrix D_i: top-left i-by-i block is 2, rest is 1."""
    D = np.ones((N, N))
    if i > 0:
        D[:i, :i] = 2
    return D


def generate_all_permutation_matrices(N):
    """
    Generates all N x N permutation matrices.

    Args:
        N (int): The size of the square matrix.

    Returns:
        list: A list of all N x N permutation matrices (NumPy arrays).
    """
    if N <= 0:
        return []

    matrices = []
    # Generate all permutations of indices (0 to N-1)
    for p in itertools.permutations(range(N)):
        # Create an N x N zero matrix
        matrix = np.zeros((N, N), dtype=int)
        # Place 1s according to the current permutation
        for i, j in enumerate(p):
            matrix[i, j] = 1
        matrices.append(matrix)
    return matrices


def project_to_doubly_stochastic(P, max_iter=100, tol=1e-6):
    """
    Iterative Sinkhorn-Knopp-like projection onto the Birkhoff polytope (doubly stochastic matrices).
    This is an approximation but works well in PGD for this non-convex problem.
    """
    p_proj = np.copy(P)

    # Ensure non-negativity first
    p_proj[p_proj < 0] = 0

    # Iteratively normalize rows and columns
    for _ in range(max_iter):
        row_sums = p_proj.sum(axis=1, keepdims=True)
        # Handle zero division for robustness
        row_sums[row_sums == 0] = 1e-10
        p_proj = p_proj / row_sums  # Row normalization

        col_sums = p_proj.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1e-10
        p_proj = p_proj / col_sums  # Column normalization

        # Check convergence (optional, using max diff from 1 for row/col sums)
        max_diff = np.max(np.abs(p_proj.sum(axis=0) - 1)) + np.max(np.abs(p_proj.sum(axis=1) - 1))
        if max_diff < tol:
            break

    # Final small adjustments to ensure sums are exactly 1 due to floating point
    p_proj = p_proj / p_proj.sum(axis=1, keepdims=True)
    p_proj = p_proj / p_proj.sum(axis=0, keepdims=True)

    return p_proj


def obj_and_grad(P, G, gamma, alpha):
    N = G.shape[0]
    G_perm = P @ G @ P.T  # the permuted G
    print("G_perm", G_perm)
    H = np.linalg.inv(np.eye(N) - gamma * G_perm)
    P_alpha = P @ alpha

    JP = 0
    GP = 0
    Q = np.ones((N, N))
    for i in range(N):
        # record objective and gradient
        he = H[:,i].copy()
        hp_alpha = H @ P_alpha
        Ji_sqrt = P_alpha[i]  # the square root of J_i(P)
        JP += Ji_sqrt ** 2
        GP += 2 * Ji_sqrt * (
            2*gamma*G@P.T@(he[:, np.newaxis] * Q * hp_alpha[np.newaxis, :])
            + np.outer(alpha, he)
        )

        # calculate the next H matrix update
        if i == N - 1:
            break

        g = G_perm[:,i].copy()
        g[i] /= 2
        g[i+1:] = 0  # zero out the lower part
        hg = H @ g
        M = np.asarray([
            [-1/gamma + hg[i], -he[i]],
            [-np.inner(g, hg), -1/gamma + hg[i]]
        ])
        M /= ((-1/gamma + hg[i])**2 - he[i] * np.inner(g, hg))  # the inverse of the middle matrix
        delta = np.column_stack((hg, he)) @ (M @ np.column_stack((he, hg)).T)

        # variable updates
        H = H - delta
        Q[:i+1, :i+1] = 2  # the mask matrix Q_i

        # sanity check
        if N <= 5 and not np.allclose(H, np.linalg.inv(np.eye(N) - gamma * np.multiply(Q, G_perm)), atol=1e-3):
            print("Updated H", H)
            print("Expected inverse", np.linalg.inv(np.eye(N) - gamma * np.multiply(Q, G_perm)))
            print("H inverse", np.linalg.inv(H))
            ei = np.eye(N)[:, i]
            print("H inverse recursion", np.linalg.inv(H + delta) - gamma * (np.outer(g, ei) + np.outer(ei, g)))
            print("Expected H inverse", np.eye(N) - gamma * np.multiply(Q, G_perm))
            raise ValueError('Inverse update error')

    return JP, GP


class AdamOptimizer:
    """
    Implements the Adam (Adaptive Moment Estimation) optimization algorithm.

    Adam combines the advantages of AdaGrad (which works well with sparse gradients)
    and RMSProp (which handles non-stationary objectives).
    """

    def __init__(self, N, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initializes the Adam optimizer.

        :param learning_rate: The step size (alpha).
        :param beta1: Exponential decay rate for the first moment estimates (momentum).
        :param beta2: Exponential decay rate for the second moment estimates (uncentered variance).
        :param epsilon: A small constant for numerical stability (to prevent division by zero).
        """

        # Hyperparameters
        self.alpha = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # State variables
        self.t = 0  # Time step
        self.m = np.zeros((N, N))  # First moment vector (momentum)
        self.v = np.zeros((N, N))  # Second moment vector (uncentered variance)

    def update(self, params, gradient):
        """
        Performs a single Adam update step using the provided gradient.

        :param gradient: The gradient vector of the loss function w.r.t. the parameters.
        :return: The updated parameter values.
        """
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params - self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)


def solve_pgd(G, initial_P, gamma, alpha, beta = 0.9, learning_rate=1e-2, max_iterations=5000, tol=1e-6, verbose=True):
    """
    Performs Projected Gradient Descent to minimize J(P).
    """
    N = G.shape[0]
    P = initial_P.copy()
    J_history = []

    print(f"Starting PGD for N={N}. Initial P is a random doubly stochastic matrix.")
    optimizer = AdamOptimizer(
        N=N,
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999
    )

    for iter_count in range(max_iterations):
        # Pass the new alpha vector to the objective function
        J_P, nabla_J_P = obj_and_grad(P, G, gamma, alpha)
        J_history.append(J_P)

        # Check for convergence
        if iter_count > 0 and abs(J_history[-1] - J_history[-2]) < tol:
            print(f"Convergence achieved at iteration {iter_count}.")
            break

        # Gradient Descent Step
        P_next = optimizer.update(P, nabla_J_P)

        # Projection Step
        P = project_to_doubly_stochastic(P_next)

        if verbose and iter_count % 100 == 0:
            grad_norm = np.linalg.norm(nabla_J_P)
            print(f"Iteration {iter_count:04d} | J(P) = {J_P:.6f} | Gradient Norm: {grad_norm:.6e}")

    print(f"\nOptimization Finished.")
    print(f"Final J(P): {J_history[-1]:.6f}")

    return P, J_history


def leave_one_out_heuristic(N, G, gamma, alpha):
    # This is also the "pivotal" heuristic as agent taxes in pivotal mechanism
    # is the difference between leave-one-out utility and social optimum.
    leave_one_out_utility = []
    for i in range(N):
        Q_loo = np.ones((N, N)) * 2
        Q_loo[i, :] = 1
        Q_loo[:, i] = 1
        H_loo = np.linalg.inv(np.eye(N) - gamma * np.multiply(Q_loo, G))
        u_loo = H_loo @ alpha
        leave_one_out_utility.append(u_loo[i])

    # return the sorted indices based on leave-one-out utility
    return np.argsort(leave_one_out_utility)


def greedy_welfare_heuristic(N, G, gamma, alpha):
    # find the greedy ordering based on marginal contribution to the social welfare
    remaining_agents = set(range(N))
    ordering = []
    while remaining_agents:
        best_agent = None
        best_increase = -np.inf
        for agent in remaining_agents:
            current_ordering = ordering + [agent]
            Q_greedy = np.ones((N, N))
            for i in range(len(current_ordering)):
                for j in range(len(current_ordering)):
                    Q_greedy[current_ordering[i], current_ordering[j]] = 2

            H_greedy = np.linalg.inv(np.eye(N) - gamma * np.multiply(Q_greedy, G))
            u_greedy = H_greedy @ alpha
            social_welfare = np.sum(u_greedy)
            if social_welfare > best_increase:
                best_increase = social_welfare
                best_agent = agent

        ordering.append(best_agent)
        remaining_agents.remove(best_agent)

    return ordering


def nash_heuristic(N, G, gamma, alpha):
    ...


def social_optimum(N, G, gamma, alpha):
    return np.inner(alpha, np.linalg.inv(np.eye(N) - 2*gamma*G) @ alpha)


def main(N, G, alpha, gamma, verbose):
    repeat_num = 10
    exp_history = []
    for _ in trange(repeat_num, desc="Runs"):
        P_initial_random = np.random.rand(N, N)
        P_initial = project_to_doubly_stochastic(P_initial_random)
        LR = 2e-6  # Adjusted learning rate, often needed when the gradient grows in complexity
        MAX_ITER = 10000
        P_final, J_history = solve_pgd(G, P_initial, gamma, alpha,
                                       learning_rate=LR, max_iterations=MAX_ITER, verbose=verbose)
        exp_history.append((P_final, J_history))

        if verbose:
            print("\n--- Final Optimized Doubly Stochastic Matrix P ---")
            np.set_printoptions(precision=4, suppress=True)
            print(P_final)

            # Verification of Doubly Stochastic Constraint
            print("\nVerification:")
            print(f"Row sums: {P_final.sum(axis=1)}")
            print(f"Col sums: {P_final.sum(axis=0)}")

    min_J = min([(P, history[-1]) for P, history in exp_history], key=lambda x: x[1])
    print("Minimum objective value over all runs:", min_J[1])
    print("Corresponding P matrix:\n", min_J[0])


def single_eval(N, G, alpha, gamma, P):
    J_P, _ = obj_and_grad(P, G, gamma, alpha)
    return J_P


def main_eval(N, G, alpha, gamma):
    permutation_matrices = generate_all_permutation_matrices(N)
    for idx, P in enumerate(permutation_matrices):
        print("Permutation Matrix:", P)
        J_P = single_eval(N, G, alpha, gamma, P)
        print(f"Permutation Matrix {idx+1}/{len(permutation_matrices)}: J(P) = {J_P:.6f}")


# --- 4. Execution ---
if __name__ == "__main__":
    np.random.seed(99)  # New seed for alpha
    N = 20  # Number of agents/nodes
    G = create_adjacency_matrix(N)
    print('G adjacency matrix:\n', G)
    gamma = 0.01  # Interaction strength parameter
    alpha = np.random.rand(N) * 10
    print('alpha', alpha)
    verbose = False

    # # RUN!: Calculate the approximation
    # main(N, G, alpha, gamma, verbose)

    # RUN2: Evaluate all permutation matrices
    main_eval(N, G, alpha, gamma)

    # # RUN3: Evaluate a particular matrix P
    # test_P = np.asarray([
    #      [0.23500973, 0.20902698, 0.19163777, 0.19056326, 0.17376227],
    #      [0.28845243, 0.14990515, 0.,         0.34769386, 0.2139486 ],
    #      [0.24473727, 0.36197843, 0.27475577, 0.11852858, 0.        ],
    #      [0.1351126,  0.08125559, 0.16509261, 0.19518997, 0.42334918],
    #      [0.09668798, 0.19783385, 0.36851385, 0.14802433, 0.18893995]
    # ])
    # J_P = single_eval(N, G, alpha, gamma, test_P)
    # print(f"Test P: J(P) = {J_P:.6f}")

    # # RUN4: Compute the pivotal heuristic orderings
    # loo_order = leave_one_out_heuristic(N, G, gamma, alpha)
    # print("Leave-One-Out Heuristic Ordering:", loo_order.tolist())
    # J_P = single_eval(N, G, alpha, gamma, np.eye(N)[loo_order])  # 127.392063
    # print(f"Leave-One-Out Heuristic: J(P) = {J_P:.6f}")
    #
    # # RUN5: Compute the greedy welfare heuristic orderings
    # greedy_order = greedy_welfare_heuristic(N, G, gamma, alpha)
    # print("Greedy Welfare Heuristic Ordering:", greedy_order)
    # J_P = single_eval(N, G, alpha, gamma, np.eye(N)[greedy_order])
    # print(f"Greedy Welfare Heuristic: J(P) = {J_P:.6f}")
