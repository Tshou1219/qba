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
    neighbor = None
    count = np.array([])

    def __init__(self, edges: List[Tuple[int, int]]) -> None:
        self.initilize_graph(edges)

    def initilize_graph(self, edges):
        self.G = DiGraph(edges+[(b, a) for (a, b) in edges])
        self.curved_edge = [edge for edge in self.G.edges()]
        self.curved_weight = [1.0 / len(self.curved_edge)
                              for _ in range(len(self.curved_edge))]
        self.neighbor = [[i for i in self.G.neighbors(
            j)] for j in range(self.G.number_of_nodes())]

    def grover(self, max, path: List[Tuple[int, float]]):
        flowed = len(path) != 0
        deg = [int(d/2) for _, d in self.G.degree()]
        if flowed:
            for p in path:
                deg[p[0]] += 1
            grover_matrix = self.make_grover_matrix(len(path))
            flowed_weight = [flowed for flowed in zip(*path)]
            beta_in = flowed_weight[1]
            beta_out = grover_matrix@beta_in
        for n in range(max):
            weight_copy = [0.0 for _ in range(len(self.curved_edge))]
            out = [p[1]*(2/(deg[p[0]])-1) for p in path]
            for i, (weight, edges) in enumerate(zip(self.curved_weight, self.curved_edge)):
                if flowed:
                    for p in path:
                        if edges[0] == p[0]:
                            weight_copy[i] += p[1]*(2/(deg[edges[0]]))
                    for p in list(flowed_weight[0]):
                        if self.curved_edge[i][1] == p:
                            out[list(flowed_weight[0]).index(p)
                                ] += weight*(2/(deg[p]))

                if weight == 0:
                    continue
                for neighborhood in self.neighbor[edges[1]]:
                    j = self.curved_edge.index((edges[1], neighborhood))
                    if edges[0] == neighborhood:
                        weight_copy[j] += weight * \
                            ((2/(deg[edges[1]]))-1)
                    else:
                        weight_copy[j] += weight * \
                            (2/(deg[edges[1]]))
            if flowed:
                # if n == max-1 and np.linalg.norm(out-beta_out, ord=2) < 0.01:
                #     print(np.linalg.norm(np.array(out)-np.array(beta_out), ord=2))
                #     break
                # if self.check_convergence(np.linalg.norm(np.array(out)-np.array(beta_out)), 0.01):
                #     print(self.G.number_of_nodes()-3, n)
                #     break
                # if max-10 < n and n < max:
                #     print(n, np.linalg.norm(np.array(
                #         self.curved_weight) - np.array(weight_copy), ord=2))
                #     print(np.array(weight_copy))
                ###########
                if n == max-1:
                    break
                if self.check_convergence(np.linalg.norm(np.array(self.curved_weight)-np.array(weight_copy), ord=2), 0.01):
                    #print(self.G.number_of_nodes()-3, n)
                    self.count = np.append(self.count, n)
                    break
            self.curved_weight = weight_copy
        self.curved_edge_labels = {edge: round(weight, 2) for edge,
                                   weight in zip(self.curved_edge, self.curved_weight)}

    def ba_run(self, m, N):
        for i in range(self.G.number_of_nodes(), N):
            self.G.add_nodes_from([i])
            nodea = np.array(self.G.nodes())
            dega = np.array(self.G.to_undirected().degree())[:, 1]
            for _ in range(m):
                while True:
                    new = int(
                        str(random.choices(nodea, dega/np.sum(dega)))[1:].rstrip(']'))
                    if self.G.has_edge(i, new) or i == new:
                        continue
                    self.G.add_edges_from([(i, new)])
                    break

    def qba_run(self, m, path: List[Tuple[int, float]]):
        nodea = np.array(self.G.nodes())
        weight_array = [0.0 for _ in range(self.G.number_of_nodes())]
        for edges, weight in zip(self.curved_edge, self.curved_weight):
            weight_array[edges[1]] += weight**2
        for p in path:
            weight_array[p[0]] += (p[1])**2
        new_node = self.G.number_of_nodes()
        self.G.add_nodes_from([new_node])
        self.neighbor.append([])
        for _ in range(m):
            while True:
                selected = int(str(random.choices(
                    nodea, weight_array/np.sum(weight_array)))[1:].rstrip(']'))
                if self.G.has_edge(new_node, selected) or new_node == selected:
                    continue
                self.G.add_edges_from([(new_node, selected)])
                self.G.add_edges_from([(selected, new_node)])
                break
            self.curved_edge.append((new_node, selected))
            self.curved_weight.append(self.curved_weight[-1])
            self.curved_edge.append((selected, new_node))
            self.curved_weight.append(self.curved_weight[-1])
            self.neighbor[new_node].append(selected)
            self.neighbor[selected].append(new_node)
            self.curved_weight = [1.0 for _ in range(len(self.curved_edge))]

    def complete_graph(self, n: int):
        node = list(map(int, range(n)))
        edges = [(a, b) for idx, a in enumerate(node) for b in node[idx + 1:]] + \
            [(b, a) for idx, a in enumerate(node) for b in node[idx + 1:]]
        self.initilize_graph(edges)

    def make_grover_matrix(self, n):
        grover_matrix = np.full((n, n), 2/n)
        for i in range(n):
            grover_matrix[i][i] -= 1.0
        return grover_matrix

    def check_convergence(self, norm, epsilon) -> bool:
        if norm < epsilon:
            return True
        else:
            return False

    def make_count_histgram(self):
        plt.figure(facecolor="azure", edgecolor="coral")
        plt.bar([i for i, _ in enumerate(self.count)], self.count)
        plt.title("number of run")
        plt.show()

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

    def deg_plot(self):
        degree_sequence = sorted(
            (d for n, d in self.G.to_undirected().degree()), reverse=True)
        dmax = max(degree_sequence)

        fig = plt.figure("Degree of a random graph", figsize=(8, 8))

        axgrid = fig.add_gridspec(5, 4)

        ax0 = fig.add_subplot(axgrid[0:3, :])
        Gcc = self.G.to_undirected().subgraph(
            sorted(nx.connected_components(self.G.to_undirected()), key=len, reverse=True)[0])
        pos = nx.spring_layout(Gcc, seed=10396953)
        nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
        nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
        ax0.set_title("Connected components of G")
        ax0.set_axis_off()

        ax1 = fig.add_subplot(axgrid[3:, :2])
        ax1.plot(degree_sequence, "b-", marker="o")
        ax1.set_title("Degree Rank Plot")
        ax1.set_ylabel("Degree")
        ax1.set_xlabel("Rank")

        ax2 = fig.add_subplot(axgrid[3:, 2:])
        ax2.bar(*np.unique(degree_sequence, return_counts=True))
        ax2.set_title("Degree histogram")
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("# of Nodes")

        fig.tight_layout()
        plt.show()
