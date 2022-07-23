import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx import DiGraph
import random
from typing import List, Tuple
import label


class Grover():
    G = DiGraph()
    curved_edge = None
    curved_edge_labels = None
    curved_weight = None

    def __init__(self, edges: List[Tuple[int, int]]) -> None:
        self.G = DiGraph(edges+[(b, a) for (a, b) in edges])

    def run_grover(self, n):
        deg = [int(d/2) for _, d in self.G.degree()]
        self.curved_edge = [edge for edge in self.G.edges(
        ) if reversed(edge) in self.G.edges()]
        self.curved_weight = [0.0 for _ in range(len(self.curved_edge))]
        # 初期状態の定義
        self.curved_weight[3] = 1.0

        for n in range(n):
            weight_copy = [0.0 for _ in range(len(self.curved_edge))]
            for i, (weight, edges) in enumerate(zip(self.curved_weight, self.curved_edge)):
                if weight == 0:
                    continue
                for j, edges in enumerate(self.curved_edge):
                    if edges[0] == self.curved_edge[i][1]:
                        if edges[1] == self.curved_edge[i][0]:
                            weight_copy[j] += weight * \
                                ((2/(deg[edges[0]]))-1)
                        else:
                            weight_copy[j] += weight*(2/(deg[edges[0]]))
            self.curved_weight = weight_copy
        self.curved_edge_labels = {edge: weight for edge,
                                   weight in zip(self.curved_edge, self.curved_weight)}

    def run_flowed_grover(self, n, path: List[Tuple[int, float]]):
        deg = [int(d/2) for _, d in self.G.degree()]
        for p in path:
            deg[p[0]] += 1
        self.curved_edge = [edge for edge in self.G.edges(
        ) if reversed(edge) in self.G.edges()]
        self.curved_weight = [0.0 for _ in range(len(self.curved_edge))]
        # 初期状態の定義
        # self.curved_weight[3] = 1.0
        # self.curved_weight[4] = 1/np.sqrt(2)

        grover_matrix = np.full((len(path), len(path)), 2/(len(path)))
        for i in range(len(path)):
            grover_matrix[i][i] -= 1.0
        div = [flowed_weight for flowed_weight in zip(*path)]
        beta_in = div[1]
        beta_out = grover_matrix@beta_in
        # print(beta_out)

        for n in range(n):
            weight_copy = [0.0 for _ in range(len(self.curved_edge))]
            out = [p[1]*(2/(deg[p[0]])-1) for p in path]
            for i, (weight, edges) in enumerate(zip(self.curved_weight, self.curved_edge)):
                for p in path:
                    if edges[0] == p[0]:
                        weight_copy[i] += p[1]*(2/(deg[edges[0]]))
                if weight == 0:
                    continue
                for p in list(div[0]):
                    if self.curved_edge[i][1] == p:
                        out[list(div[0]).index(p)] += weight*(2/(deg[p]))
                for j, edges in enumerate(self.curved_edge):
                    if edges[0] == self.curved_edge[i][1]:
                        if edges[1] == self.curved_edge[i][0]:
                            weight_copy[j] += weight * \
                                ((2/(deg[edges[0]]))-1)
                        else:
                            weight_copy[j] += weight*(2/(deg[edges[0]]))
            # 収束判定
            if np.linalg.norm(out-beta_out, ord=2) < 0.001:
                # print(n)
                break
            ###
            self.curved_weight = weight_copy
        # print(out)
        self.curved_edge_labels = {edge: weight for edge,
                                   weight in zip(self.curved_edge, self.curved_weight)}

    # def make_grover_matrix(self, n):
    #     grover_matrix = np.full((n, n), 2/n)
    #     for i in range(n):
    #         grover_matrix[i][i] -= 1
    #     print(grover_matrix@[1, 1, 0])

    def qba_run(self, m, path: List[Tuple[int, float]]):
        nodea = np.array(self.G.nodes())
        weight_array = [0.0 for _ in range(self.G.number_of_nodes())]
        #print(self.curved_edge, self.curved_weight)
        for edges, weight in zip(self.curved_edge, self.curved_weight):
            #print(edges[1], self.curved_weight[edges[1]])
            weight_array[edges[1]] += weight**2
        for p in path:
            # print(p[1])
            weight_array[p[0]] += (p[1])**2
        # print(weight_array)
        new_node = self.G.number_of_nodes()
        self.G.add_nodes_from([new_node])
        for _ in range(m):
            while True:
                select = int(str(random.choices(
                    nodea, weight_array/np.sum(weight_array)))[1:].rstrip(']'))
                # print(select)
                if self.G.has_edge(new_node, select) or new_node == select:
                    continue
                self.G.add_edges_from([(new_node, select)])
                self.G.add_edges_from([(select, new_node)])
                break

    def complete_graph(self, n: int) -> DiGraph:
        node = list(map(int, range(n)))
        self.G = DiGraph()
        self.G.add_nodes_from(node)
        self.G.add_edges_from(
            [(a, b) for idx, a in enumerate(node) for b in node[idx + 1:]])
        self.G.add_edges_from(
            [(b, a) for idx, a in enumerate(node) for b in node[idx + 1:]])

    def plot(self):

        nx.draw(self.G.to_undirected(), with_labels=True)
        plt.show()

    def arc_plot(self):
        pos = nx.spring_layout(self.G, seed=5)
        fig, ax = plt.subplots()
        nx.draw_networkx_nodes(self.G, pos, ax=ax)
        nx.draw_networkx_labels(self.G, pos, ax=ax)
        nx.draw_networkx_edges(
            self.G, pos, ax=ax, edgelist=self.curved_edge, connectionstyle=f'arc3, rad = 0.25')
        label.my_draw_networkx_edge_labels(
            self.G, pos, ax=ax, edge_labels=self.curved_edge_labels, rotate=False, rad=0.25)
        #print(self.curved_edge, self.curved_weight)
