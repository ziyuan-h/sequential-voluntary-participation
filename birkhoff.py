import numpy as np


def _find_permutation_matching(D, row_index, N, matching, used_cols,
                               epsilon=1e-9):
  """
  Internal recursive function to find a perfect matching (a permutation)
  using Depth First Search on the non-zero entries of matrix D.

  Args:
      D (np.ndarray): The current working doubly stochastic matrix (or sub-stochastic matrix).
      row_index (int): The current row being processed (0 to N-1).
      N (int): The dimension of the square matrix.
      matching (dict): Stores the current row -> column assignments (r -> c).
      used_cols (set): Stores columns that have already been matched.
      epsilon (float): Tolerance for treating an entry as zero.

  Returns:
      bool: True if a perfect matching is found from this state, False otherwise.
  """
  # Base case: All rows have been successfully matched.
  if row_index == N:
    return True
  
  # Try matching the current row (row_index) with all possible columns
  for col_index in range(N):
    # Check if D[r, c] is non-zero (within epsilon tolerance)
    # AND if the column has not been used yet.
    if D[row_index, col_index] > epsilon and col_index not in used_cols:
      # 1. Tentatively assign the match
      matching[row_index] = col_index
      used_cols.add(col_index)
      
      # 2. Recurse to the next row
      if _find_permutation_matching(D, row_index + 1, N, matching, used_cols,
                                    epsilon):
        return True  # Success
      
      # 3. Backtrack: If recursion fails, un-assign and un-use the column
      del matching[row_index]
      used_cols.remove(col_index)
  
  # If no column works for the current row, backtrack (failure)
  return False


def bvn_decomposition(D_in, epsilon=1e-9) -> list:
  """
  Performs the Birkhoff-von Neumann decomposition of a doubly stochastic matrix.

  Any doubly stochastic matrix D can be written as a convex combination of
  permutation matrices P_i: D = sum(lambda_i * P_i).

  Args:
      D_in (np.ndarray): The square doubly stochastic matrix to decompose.
      epsilon (float): Numerical tolerance for zero comparisons.

  Returns:
      list: A list of tuples (lambda_i, P_i), where lambda_i is the coefficient
            and P_i is the corresponding permutation matrix.
  """
  # 1. Validation and Initialization
  D = D_in.copy()
  N = D.shape[0]
  
  if D.shape[0] != D.shape[1]:
    raise ValueError("Input matrix must be square.")
  if not np.all(D >= -epsilon):
    raise ValueError("Matrix must have non-negative entries.")
  if not np.allclose(D.sum(axis=0), 1.0) or not np.allclose(D.sum(axis=1), 1.0):
    # Allow small deviation from 1.0 due to floating point arithmetic
    print(
      "Warning: Input matrix might not be perfectly doubly stochastic (row/column sums not exactly 1).")
  
  decomposition = []
  
  # 2. Iterative Decomposition
  while not np.allclose(D, 0.0, atol=epsilon):
    # --- Step A: Find a Perfect Matching (Permutation) ---
    matching = {}
    used_cols = set()
    
    # Call the recursive DFS function to find the permutation based on non-zero entries
    success = _find_permutation_matching(D, 0, N, matching, used_cols, epsilon)
    
    if not success:
      # This should not happen for a numerically valid doubly stochastic matrix,
      # but is a safe guard against numerical instability or non-stochastic input.
      print("Error: Could not find a perfect matching. Decomposition stopped.")
      break
    
    # Convert the matching dictionary into a permutation matrix P
    P = np.zeros_like(D)
    for r, c in matching.items():
      P[r, c] = 1.0
    
    # --- Step B: Determine the Coefficient (Lambda) ---
    # Lambda is the minimum value in D along the path defined by P
    # We only look at entries where P[r, c] == 1
    lambda_i = np.min([D[r, c] for r, c in matching.items()])
    
    # Ensure lambda_i is non-negative, accounting for small numerical errors
    lambda_i = max(0.0, lambda_i)
    
    # --- Step C: Update and Store ---
    decomposition.append((lambda_i, P))
    
    # Update the matrix D: D = D - lambda_i * P
    D = D - lambda_i * P
    
    # Due to floating point arithmetic, very small negative numbers might appear.
    # Set them explicitly to zero.
    D[D < epsilon] = 0.0
  
  # 3. Normalize the Coefficients (Lambdas)
  # The sum of all lambdas should be 1.0.
  total_lambda = sum(lam for lam, _ in decomposition)
  
  # Final normalized decomposition
  normalized_decomposition = [
    (lam / total_lambda, P)
    for lam, P in decomposition
  ]
  
  print(
    f"Decomposition successful! Found {len(normalized_decomposition)} components.")
  return normalized_decomposition


if __name__ == "__main__":
  
  # --- Example Usage ---
  
  # Example 1: A 3x3 doubly stochastic matrix
  D1 = np.array([
    [0.7, 0.3, 0.0],
    [0.0, 0.6, 0.4],
    [0.3, 0.1, 0.6]
  ])
  
  print("--- Decomposition of Example 1 (D1) ---")
  print("Original Matrix D1:\n", D1)
  result1 = bvn_decomposition(D1)
  
  print("\nDecomposition Components (lambda, P_i):")
  total_sum_matrix = np.zeros_like(D1)
  for i, (lam, P) in enumerate(result1):
    print(f"\nComponent {i + 1}: Lambda={lam:.4f}")
    print("Permutation Matrix P:\n", P)
    total_sum_matrix += lam * P
  
  print("\n--- Verification ---")
  print(f"Sum of Lambdas: {sum(lam for lam, _ in result1):.4f}")
  print("Reconstructed Matrix (Sum of lambda_i * P_i):\n", total_sum_matrix)
  
  # Example 2: A simpler 2x2 matrix
  D2 = np.array([
    [0.2, 0.8],
    [0.8, 0.2]
  ])
  
  print("\n" + "=" * 50)
  print("--- Decomposition of Example 2 (D2) ---")
  print("Original Matrix D2:\n", D2)
  result2 = bvn_decomposition(D2)
  
  print("\nDecomposition Components (lambda, P_i):")
  total_sum_matrix_2 = np.zeros_like(D2)
  for i, (lam, P) in enumerate(result2):
    print(f"\nComponent {i + 1}: Lambda={lam:.4f}")
    print("Permutation Matrix P:\n", P)
    total_sum_matrix_2 += lam * P
  
  print("\n--- Verification ---")
  print(f"Sum of Lambdas: {sum(lam for lam, _ in result2):.4f}")
  print("Reconstructed Matrix (Sum of lambda_i * P_i):\n", total_sum_matrix_2)
