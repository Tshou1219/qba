import matplotlib.pyplot as plt
import numpy as np
import random
import networkx as nx


class BAModel():
    G = None,
    m0 = 3
    m = 2
    N = 15

    def __init__(self):
        self.G = nx.Graph()
        for i in range(self.m0):
            self.G.add_nodes_from([i])
            for j in range(i):
                self.G.add_edges_from([(i, j)])

    def run(self):
        for i in range(self.m0, self.N):
            self.G.add_nodes_from([i])
            nodea = np.array(self.G.nodes())
            dega = np.array(self.G.degree())[:, 1]
            sum = np.sum(dega)
            degp = dega/sum
            for _ in range(self.m):
                while True:
                    new = int(str(random.choices(nodea, degp))[1:2])
                    if self.G.has_edge(i, new) or i == new:
                        continue
                    self.G.add_edges_from([(i, new)])
                    break

    def draw(self):
        nx.draw(self.G, with_labels=True)
        plt.show()
