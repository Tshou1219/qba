import matplotlib.pyplot as plt
import numpy as np
import random
import networkx as nx
import time


class BAModel:
    G = (None,)

    def __init__(self, m0):
        self.G = nx.Graph()
        self.G.add_nodes_from([i for i in range(m0)])
        self.G.add_edges_from([(i, j) for i in range(m0) for j in range(i)])

    def ba_run(self, m, N):
        for i in range(self.G.number_of_nodes(), N):
            self.G.add_nodes_from([i])
            nodea = np.array(self.G.nodes())
            dega = np.array(self.G.degree())[:, 1]
            for _ in range(m):
                while True:
                    new = int(
                        str(random.choices(nodea, dega / np.sum(dega)))[1:].rstrip("]")
                    )
                    # new = random.choices(nodea, dega/np.sum(dega))[0]
                    if self.G.has_edge(i, new) or i == new:
                        continue
                    self.G.add_edges_from([(i, new)])
                    break

    def draw(self):
        nx.draw(self.G, with_labels=True)
        plt.show()


class BAModel_Opt:
    def __init__(self, m0):
        self.nodes = np.array(range(m0))
        self.degs = np.array([m0 - 1] * m0)
        self.sum = (m0 - 1) * m0
        # self.edges = [(i, j) for i in range(m0) for j in range(i)]

    def ba_run(self, m, N):
        for i in range(len(self.nodes), N):
            prevs = set()
            for _ in range(m):
                while True:
                    new = np.random.choice(self.nodes, 1, p=self.degs / self.sum)[0]
                    if new in prevs:
                        continue
                    # self.edges.append((i, new))
                    self.degs[new] += 1
                    self.sum += 1
                    prevs.add(new)
                    break
            self.nodes = np.append(self.nodes, i)
            self.degs = np.append(self.degs, m)
            self.sum += m

    def draw(self):
        G = nx.Graph(list(self.edges))
        nx.draw(G, with_labels=True)
        plt.show()


time_sta = time.perf_counter()
bamodel = BAModel(4)
bamodel.ba_run(2, 10000)
time_end = time.perf_counter()
print(time_end - time_sta)
# bamodel.draw()
time_sta = time.perf_counter()
bamodel = BAModel_Opt(4)
bamodel.ba_run(1, 10000)
time_end = time.perf_counter()
print(time_end - time_sta)
# bamodel.draw()
