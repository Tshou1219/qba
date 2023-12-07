import time
import warnings
from typing import List, Tuple
import numpy as np
from networkx import DiGraph
from numba import njit, prange
from numba.core.errors import NumbaPendingDeprecationWarning
from numpy.typing import NDArray
import plot
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
np.set_printoptions(precision=3, floatmode='fixed')

import matplotlib.pyplot as plt


class QBA():

    def __init__(self) -> None:
        self.count = []

    def initilize(self, n: int, edges: List[Tuple[int, int]]):
        self.arcs = list(DiGraph(edges+[(b, a) for (a, b) in edges]).edges())
        self.weights, self.deg = np.zeros(
            len(self.arcs)), np.zeros(n, dtype=int)
        ### set initial state
        # self.weights[0] = 1.
        # for i in range(len(self.arcs)):
        #     self.weights[i]=1.
        #######
        self.initial_state = self.weights
        for edge in edges:
            self.deg[edge[0]] += 1
            self.deg[edge[1]] += 1

    def arc_plot(self):
        label = {edge: round(weight, 2)
                 for edge, weight in zip(self.arcs, self.weights)}
        plot.plot_arc(DiGraph(self.arcs), label)

    

    def run_qba(self, n: int, max: int):
        self.times = []
        for _ in range(n):
            time_sta = time.perf_counter()
            self.grover(max)
            self.add_nodes(1)
            time_end = time.perf_counter()
            self.times.append(time_end - time_sta)
        plot.plot_time(self.times)
        plot.deg_plot(DiGraph(self.arcs))
        plot.count_histgram(self.count)

    def grover(self, n: int, end_on_conv: bool = True):

        rho = np.zeros(len(self.arcs))
        np_arc = np.array(self.arcs)
        for (node, wgt) in self.path:
            for i in np.where(np_arc[:, 0] == node)[0]:
                rho[i] = (2 / self.deg[node]) * wgt
        # print(rho)
        mat = self.make_grover_matrix()
        for i in range(n):
            next = mat @ self.weights + rho ## normal 
            # next = mat @ self.weights + rho * pow(-1 , len(self.deg)) ## change rho's parity
            # if end_on_conv and self.check_conv(next, self.weights): ## convergence check
            if end_on_conv and self.check_conv_grov(next):
                self.count.append(i)
                self.weights = next
                return
            self.weights = next
        print("not converged")
        self.count.append(n)

    def check_conv(self, new_state: NDArray, state: NDArray):
        def nmz(x): return x / np.linalg.norm(x, ord=2)
        return np.linalg.norm(nmz(new_state) - nmz(state), ord=2) < 0.01

    def check_conv_grov(self, next: NDArray):
        n = len(self.path)
        grov_mat = np.full((n, n), 2 / n) - np.identity(n)
        beta_out = grov_mat @ list(zip(*self.path))[1]
        out = np.array([p[1] * (2 / self.deg[p[0]] - 1) for p in self.path])
        np_arc = np.array(self.arcs)
        for path_id, (node, _) in enumerate(self.path):
            for i in np.where(np_arc[:, 1] == node)[0]:
                out[path_id] += (2 / self.deg[node]) * next[i]
        return np.linalg.norm(out - beta_out, ord=2) < 0.001

    def add_nodes(self, m: int):
        probs = np.zeros(len(self.deg))
        for edges, weight in zip(self.arcs, self.weights):
            probs[edges[1]] += weight**2
        for p in self.path:
            probs[p[0]] += (p[1])**2
        prevs = set()
        for _ in range(m):
            while True:
                selected = np.random.choice(
                    np.array(range(len(self.deg))), 1, p=probs/np.sum(probs))[0]
                if selected in prevs:
                    continue
                prevs.add(selected)
                break
            self.arcs.extend([(len(self.deg), selected),
                             (selected, len(self.deg))])
            # self.weights=np.append(self.weights,[self.weights[-1]]*2)
            self.weights = np.ones(len(self.arcs))
            self.deg[selected] += 1
        self.deg = np.append(self.deg, m)

    def make_grover_matrix(self):
        return make_grover_matrix(self.deg, np.array(self.arcs))
    
    ################

    def non_flow_grover(self, n: int, v_prob, num_pin = 0, inverse=False):
        mat=self.make_grover_matrix()
        # print(mat)
        ## 最後だけ反転させる
        # mat = mat @ self.grover_inv(num_pin)
        if inverse:
            mat = self.grover_c(mat,num_pin)
        # print(np.linalg.eig(mat))
        # mat=mat.T
        # print(mat)
        ##
        # for i in range(len(self.arcs)):
        #     mat[i][len(self.arcs)-2]=0
        # mat[len(self.arcs)-1][len(self.arcs)-2]=-1
        # print(mat)
        ##
        self.prob_origin=np.zeros(n)
        for i in range(n):
            probs=np.zeros(len(self.deg))
            next = mat @ self.weights
            for edges, weight in zip(self.arcs, self.weights):
                probs[edges[1]] += weight**2     ##### 重みを二乗して足す
            # print(self.arcs)
            # print(self.weights)
            # print(probs)
            for j in v_prob:
                if np.sum(probs)==0:
                    break
                else:
                    self.prob_origin[i]+=probs[j]/np.sum(probs)
            self.weights = next
        #     print(probs)
        # print(self.prob_origin)

    def non_flow_grover_arcs(self, n: int, arc_prob, num_pin = 0, inverse=False):
        mat=self.make_grover_matrix()
        # print(mat)
        ## 最後だけ反転させる
        # mat = mat @ self.grover_inv(num_pin)
        if inverse:
            mat = self.grover_c(mat,num_pin)
        self.prob_origin=np.zeros(n)
        for i in range(n):
            probs=np.zeros(len(self.deg))
            next = mat @ self.weights
            for edges, weight in zip(self.arcs, self.weights):
                probs[edges[1]] += weight**2     ##### 重みを二乗して足す
            # print(self.arcs)
            # print(self.weights)
            # print(probs)
            for j in arc_prob:
                if np.sum(probs)==0:
                    break
                else:
                    self.prob_origin[i]+=self.weights[self.arcs.index(j)]**2/np.sum(probs)
            self.weights = next
        #     print(probs)
        # print(self.prob_origin)

    def grover_inv(self, num_pin):
        mat=np.identity(len(self.arcs))
        for i in range(num_pin):
            mat[len(self.arcs)-i-1][len(self.arcs)-i-1]=-1
        # for i in range(len(self.arcs)):
        #     mat[i][len(self.arcs)-1]=-1*mat[i][len(self.arcs)-1]
        return mat

    def grover_c(self,mat,num_pin):
        for i in range(len(self.arcs)):
            for j in range(num_pin):
                mat[i][len(self.arcs)-j-2]=-1*mat[i][len(self.arcs)-j-2]
        return mat

    def plot_origin_prob(self):
        plt.figure(facecolor="azure", edgecolor="coral")
        plt.plot(self.prob_origin)
        plt.show()    

    # def add_comp_graph(self, n: int, add_v: int):
    #     node = list(map(int, range(len(self.deg),n+len(self.deg))))
    #     edges = [(a, b) for idx, a in enumerate(node) for b in node[idx + 1:]]
    #     edges.extend([(len(self.deg),add_v),(add_v,len(self.deg))])
    #     self.arcs.extend(edges)
    #     print(edges)
    #     print(self.deg)
    #     self.initilize(n+len(self.deg), self.arcs)

    def find_hamming_distance(self, x,y):
        return bin(x^y).count('1')
    #################
    
    def build_comp_graph(self, n: int, path: List[Tuple[int, float]]):
        self.initial_graph = 'complete'
        node = list(map(int, range(n)))
        edges = [(a, b) for idx, a in enumerate(node) for b in node[idx + 1:]]
        self.initilize(n, edges)
        self.path = path
        for (node, _) in self.path:
            self.deg[node] += 1
        return self
    
    def build_hypercube(self, n: int, path: List[Tuple[int, float]]):
        self.initial_graph = 'hypercybe'
        node = list(map(int, range(2**n)))
        edges = []
        for i in range(2**n): 
            for j in range(i, 2**n):
                if self.find_hamming_distance(i,j)==1:
                    edges.extend([(i,j)])
        # print(len(edges))
        self.initilize(2**n, edges)
        self.path = path
        for (node, _) in self.path:
            self.deg[node] += 1
        return self
    
    def build_comp_add_graph(self, n: int, m: int, path: List[Tuple[int, float]]):
        node = list(map(int, range(n)))
        edges = [(a, b) for idx, a in enumerate(node) for b in node[idx + 1:]]
        node = list(map(int, range(n,n+m)))
        edges.extend([(a, b) for idx, a in enumerate(node) for b in node[idx + 1:]])
        edges.extend([(n-1,n),(n,n-1)])
        self.initilize(n+m, edges)
        self.path = path
        for (node, _) in self.path:
            self.deg[node] += 1
        return self
    
    def build_barbell_graph(self, num_of_vertex:List[int], path: List[Tuple[int, float]]):
        edges=[]
        node = list(map(int, range(num_of_vertex[0])))
        edges = [(a, b) for idx, a in enumerate(node) for b in node[idx + 1:]]
        for i in range(1,len(num_of_vertex)):
            node = list(map(int, range(sum(num_of_vertex[0:i]),sum(num_of_vertex[0:i+1]))))
            edges.extend([(a, b) for idx, a in enumerate(node) for b in node[idx + 1:]])
            edges.extend([(sum(num_of_vertex[0:i])-1,sum(num_of_vertex[0:i])),(sum(num_of_vertex[0:i]),sum(num_of_vertex[0:i])-1)])
        self.initilize(sum(num_of_vertex),edges)
        for i in range(1,len(num_of_vertex)):
            self.deg[sum(num_of_vertex[0:i])-1]-=1
            self.deg[sum(num_of_vertex[0:i])]-=1
        self.path = path
        for (node, _) in self.path:
            self.deg[node] += 1
        return self
    
    def build_comp_biber_graph(self, n: int, m: int, path: List[Tuple[int, float]]):
        self.initial_graph = 'complete_bipertite'
        node = list(map(int, range(n+m)))
        edges = [(a, b) for a in range(n) for b in range(n,n+m)]
        self.initilize(n+m, edges)
        self.path = path
        for (node, _) in self.path:
            self.deg[node] += 1
        return self

    def build_circuit_graph(self, n: int, path: List[Tuple[int, float]]):
        self.initial_graph = 'circuit'
        node = list(map(int, range(n)))
        edges = [(a, a+1) for a in range(n-1)]+[(n-1, 0)]
        self.initilize(n, edges)
        self.path = path
        for (node, _) in self.path:
            self.deg[node] += 1
        return self

    def build_star_graph(self, n: int, path: List[Tuple[int, float]]):
        self.initial_graph = 'star'
        node = list(map(int, range(n)))
        edges = [(0, a) for a in range(1, n)]
        self.initilize(n, edges)
        self.path = path
        for (node, _) in self.path:
            self.deg[node] += 1
        return self

    def build_wheel_graph(self, n: int, path: List[Tuple[int, float]]):
        self.initial_graph = 'wheel'
        node = list(map(int, range(n)))
        edges = [(a, a+1) for a in range(n-2)] + [(n-2, 0)] + [(a, n-1) for a in range(n-1)]
        self.initilize(n, edges)
        self.path = path
        for (node, _) in self.path:
            self.deg[node] += 1
        return self

    def build_path_graph(self, n: int, path: List[Tuple[int, float]]):
        self.initial_graph = 'path'
        node = list(map(int, range(n)))
        edges = [(a, a+1) for a in range(n-1)]
        self.initilize(n, edges)
        self.path = path
        for (node, _) in self.path:
            self.deg[node] += 1
        return self

    def build_ladder_graph(self, n: int, path: List[Tuple[int, float]]):
        self.initial_graph = 'ladder'
        node = list(map(int, range(2*n)))
        edges = [(a, a+1) for a in range(0, 2*n, 2)]+[(a, a+2)
                                                      for a in range(2*n-2)]
        self.initilize(2*n, edges)
        self.path = path
        for (node, _) in self.path:
            self.deg[node] += 1
        return self

    def build_circular_ladder_graph(self, n: int, path: List[Tuple[int, float]]):
        self.initial_graph = 'circular_ladder'
        node = list(map(int, range(2*n)))
        edges = [(a, a+1) for a in range(0, 2*n, 2)]+[(a, a+2)
                                                      for a in range(2*n-2)] + [(0, 2*n-2), (1, 2*n-1)]
        self.initilize(2*n, edges)
        self.path = path
        for (node, _) in self.path:
            self.deg[node] += 1
        return self


@njit(parallel=True)
def make_grover_matrix(deg: NDArray, arcs: NDArray) -> NDArray:
    n = len(arcs)
    mat = np.zeros((n, n))
    for i in prange(n):
        for j in prange(n):
            if arcs[i][1] == arcs[j][0]:
                mat[j, i] = 2/(deg[arcs[i][1]]) - \
                    1 if arcs[i][0] == arcs[j][1] else 2/(deg[arcs[i][1]])
    return mat
