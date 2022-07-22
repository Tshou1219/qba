import matplotlib.pyplot as plt
import numpy as np
import random
import networkx as nx


class BAModel():
    G = None,

    def __init__(self, m0):
        self.G = nx.Graph()
        self.G.add_nodes_from([i for i in range(m0)])
        self.G.add_edges_from([(i, j) for i in range(m0)
                              for j in range(i)])

    def ba_run(self, m, N):
        for i in range(self.G.number_of_nodes(), N):
            self.G.add_nodes_from([i])
            nodea = np.array(self.G.nodes())
            dega = np.array(self.G.degree())[:, 1]
            for _ in range(m):
                while True:
                    new = int(
                        str(random.choices(nodea, dega/np.sum(dega)))[1:2])
                    if self.G.has_edge(i, new) or i == new:
                        continue
                    self.G.add_edges_from([(i, new)])
                    break

    def draw(self):
        nx.draw(self.G, with_labels=True)
        plt.show()
