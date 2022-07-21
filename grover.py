import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx import DiGraph
from typing import List, Tuple
import label


class Grover():
    G = DiGraph()
    curved_edge = None
    curved_edge_labels = None

    def __init__(self, edges: List[Tuple[int, int]]) -> None:
        self.G = DiGraph(edges+[(b, a) for (a, b) in edges])

    def run_grover(self, n):
        deg = [int(d/2) for _, d in self.G.degree()]
        self.curved_edge = [edge for edge in self.G.edges(
        ) if reversed(edge) in self.G.edges()]
        curved_weight = [0.0 for _ in range(len(self.curved_edge))]
        # 初期状態の定義
        curved_weight[3] = 1.0
        ##curved_weight[4] = 1/np.sqrt(2)
        for n in range(n):
            curved_weight_copy = [0.0 for _ in range(len(self.curved_edge))]
            for i, (weight, edges) in enumerate(zip(curved_weight, self.curved_edge)):
                if weight == 0:
                    continue
                for j, edges in enumerate(self.curved_edge):
                    if edges[0] == self.curved_edge[i][1]:
                        if edges[1] == self.curved_edge[i][0]:
                            curved_weight_copy[j] += weight * \
                                ((2/(deg[edges[0]]))-1)
                        else:
                            curved_weight_copy[j] += weight*(2/(deg[edges[0]]))
            curved_weight = curved_weight_copy
        self.curved_edge_labels = {edge: weight for edge,
                                   weight in zip(self.curved_edge, curved_weight)}

    def run_flowed_grover(self, n):
        deg = [int(d/2) for _, d in self.G.degree()]
        deg[1] += 1
        deg[2] += 1
        self.curved_edge = [edge for edge in self.G.edges(
        ) if reversed(edge) in self.G.edges()]
        curved_weight = [0.0 for _ in range(len(self.curved_edge))]
        # 初期状態の定義
        #curved_weight[3] = 1.0
        ##curved_weight[4] = 1/np.sqrt(2)
        out = 0
        for n in range(n):
            curved_weight_copy = [0.0 for _ in range(len(self.curved_edge))]
            for i, (weight, edges) in enumerate(zip(curved_weight, self.curved_edge)):
                # 1と2の点にパスをくっつける
                # print(curved_weight_copy)
                # if edges[0] == 1 or edges[0] == 2:
                #     curved_weight_copy[i] += 2*(2/(deg[edges[0]]))
                if edges[0] == 1:
                    curved_weight_copy[i] += (2/(deg[edges[0]]))
                if edges[0] == 2:
                    curved_weight_copy[i] += (2/(deg[edges[0]]))

                #
                if weight == 0:
                    continue
                for j, edges in enumerate(self.curved_edge):
                    if edges[0] == self.curved_edge[i][1]:
                        if edges[1] == self.curved_edge[i][0]:
                            curved_weight_copy[j] += weight * \
                                ((2/(deg[edges[0]]))-1)
                        else:
                            curved_weight_copy[j] += weight*(2/(deg[edges[0]]))

            curved_weight = curved_weight_copy
        out = curved_weight[0]*(2/(deg[1]))+((2/(deg[1])-1))
        self.curved_edge_labels = {edge: weight for edge,
                                   weight in zip(self.curved_edge, curved_weight)}
        print(out)

    def plot(self):
        pos = nx.spring_layout(self.G, seed=5)
        fig, ax = plt.subplots()
        nx.draw_networkx_nodes(self.G, pos, ax=ax)
        nx.draw_networkx_labels(self.G, pos, ax=ax)
        nx.draw_networkx_edges(
            self.G, pos, ax=ax, edgelist=self.curved_edge, connectionstyle=f'arc3, rad = 0.25')
        label.my_draw_networkx_edge_labels(
            self.G, pos, ax=ax, edge_labels=self.curved_edge_labels, rotate=False, rad=0.25)
