from typing import List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
from networkx import DiGraph


class ArcGraph():
    graph = DiGraph()

    def __init__(self, edges: List[Tuple[int, int]]) -> None:
        self.graph = DiGraph(edges+[(b, a) for (a, b) in edges])

    def complete_graph(self, n: int) -> DiGraph:
        node = list(map(int, range(1, n+1)))
        self.graph = DiGraph()
        self.graph.add_nodes_from(node)
        self.graph.add_edges_from(
            [(a, b) for idx, a in enumerate(node) for b in node[idx + 1:]])
        self.graph.add_edges_from(
            [(b, a) for idx, a in enumerate(node) for b in node[idx + 1:]])

    def plot(self):
        # pos = nx.spring_layout(self.graph)
        # plt.figure()
        nx.draw(self.graph, with_labels=True,
                connectionstyle='arc3, rad = 0.1')
        # nx.draw_networkx_edge_labels(
        #     self.graph, pos,
        #     edge_labels={edge: "0" for edge in list(self.graph.edges())},
        # )
        plt.show()
