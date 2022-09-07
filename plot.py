from typing import List
import networkx as nx
import matplotlib.pyplot as plt
import label
import numpy as np


def plot_time(times: List[float]):
    plt.figure(facecolor="azure", edgecolor="coral")
    plt.title("times")
    plt.xlabel("total:{}".format(sum(times)))
    plt.bar([i for i, _ in enumerate(times)], times)
    plt.show()


def plot_arc(G: nx.DiGraph, labels):
    pos = nx.spring_layout(G, seed=5)
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(
        G, pos, ax=ax, edgelist=G.edges(), connectionstyle=f'arc3, rad = 0.25')
    label.my_draw_networkx_edge_labels(
        G, pos, ax=ax, edge_labels=labels, rotate=False, rad=0.25)


def deg_plot(G: nx.DiGraph):
    degree_sequence = sorted(
        (d for n, d in G.to_undirected().degree()), reverse=True)
    dmax = max(degree_sequence)

    fig = plt.figure("Degree of a random graph",
                     facecolor="azure", edgecolor="coral", figsize=(8, 8))

    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    Gcc = G.to_undirected().subgraph(
        sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)[0])
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


def count_histgram(count):
    plt.figure(facecolor="azure", edgecolor="coral")
    plt.bar([i for i, _ in enumerate(count)], count)
    plt.title("number of run")
    plt.show()
