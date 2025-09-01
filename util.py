import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def generate_random_symmetric_binary_matrix(N: int, edge_pr: float = 0.5) -> np.ndarray:
  """Generates a random symmetric matrix of size N with values 0 or 1."""
  random_matrix = np.random.rand(N, N)
  binary_matrix = (random_matrix > edge_pr).astype(int)
  symmetric_binary_matrix = (binary_matrix + binary_matrix.T > 0).astype(int)
  np.fill_diagonal(symmetric_binary_matrix, 0)
  return symmetric_binary_matrix


def draw_network(g):
  G = nx.from_numpy_array(g)
  nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
  plt.title("Graph from Adjacency Matrix")
  plt.show()


def get_one_divide_spectral_radius(g: np.ndarray) -> float:
    return 1 / np.max(np.abs(np.linalg.eigvals(g)))