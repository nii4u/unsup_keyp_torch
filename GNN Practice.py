import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print(data)

### How to create a simple GCN

from collections import defaultdict

class Node:
def __init__(self, id, x, y, mu):
self.id = id
self.x = x;
self.y = y
self.mu = mu
def __hash__(self):
return hash((self.id))

class Graph:
def __init___(self):
self.nodes = set()
self.edges = defaultdict(set)
def add_add(self, source, target):
self.edges[source].add(target)
self.edges[target].add(source)
self.edges.add((source, target))
self.nodes.add(source)
self.nodes.add(target)
def get_neighbors(self, node):
return self.edges[node]