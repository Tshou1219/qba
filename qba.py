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

class QBA():

    def __init__(self) -> None:
        self.count = []

    def initilize(self, n:int, edges: List[Tuple[int, int]]):
        self.arcs = list(DiGraph(edges+[(b, a) for (a, b) in edges]).edges()) 
        self.weights, self.deg = np.ones(len(self.arcs)), np.zeros(n,dtype=int)
        for edge in edges:
            self.deg[edge[0]] += 1; self.deg[edge[1]] += 1

    def arc_plot(self):
        label = {edge: round(weight, 2) for edge, weight in zip(self.arcs, self.weights)}
        plot.plot_arc(DiGraph(self.arcs), label)

    def build_comp_graph(self,n:int, path: List[Tuple[int, float]]):
        node = list(map(int, range(n)))
        edges = [(a, b) for idx, a in enumerate(node) for b in node[idx + 1:]] 
        self.initilize(n,edges)
        self.path=path
        for (node,_) in self.path:
            self.deg[node] += 1
        return self

    def run_qba(self, n:int, max:int):
        times = []
        for _ in range(n):
            time_sta = time.perf_counter()
            self.grover(max)
            self.add_nodes(2)
            time_end = time.perf_counter()
            times.append(time_end - time_sta)
        plot.plot_time(times)
        plot.deg_plot(DiGraph(self.arcs))
        plot.count_histgram(self.count)

    def grover(self, n:int, end_on_conv:bool=True):    

        rho =np.zeros(len(self.arcs))
        np_arc = np.array(self.arcs)
        for (node, wgt) in self.path:
            for i in np.where(np_arc[:,0]==node)[0]:
                rho[i]=(2 / self.deg[node]) * wgt
                
        mat = self.make_grover_matrix()
        for i in range(n):
            next = mat @ self.weights + rho    
            if end_on_conv and self.check_conv(next,self.weights):
                self.count.append(i)
                self.weights = next
                return 
            self.weights = next
        print("not converged")
        self.count.append(n)

    def check_conv(self, new_state:NDArray, state:NDArray):
        nmz = lambda x:x / np.linalg.norm(x,ord=2)
        return np.linalg.norm(nmz(new_state) - nmz(state), ord=2) < 0.01
    
    def check_conv_grov(self, next:NDArray):
        n = len(self.path)
        grov_mat = np.full((n, n), 2 / n) - np.identity(n)
        beta_out = grov_mat @ list(zip(*self.path))[1]
        out = np.array([p[1] * (2 / self.deg[p[0]] - 1) for p in self.path])
        np_arc = np.array(self.arcs)
        for path_id, (node, _) in enumerate(self.path):
            for i in np.where(np_arc[:,1]==node)[0]:
                out[path_id] += (2 / self.deg[node]) * next[i]
        return np.linalg.norm(out - beta_out, ord=2) < 0.01
    
    def add_nodes(self, m:int):
        probs = np.zeros(len(self.deg))
        for edges, weight in zip(self.arcs, self.weights):
            probs[edges[1]] += weight**2
        for p in self.path:
            probs[p[0]] += (p[1])**2
        prevs = set()
        for _ in range(m):
            while True:
                selected = np.random.choice(np.array(range(len(self.deg))), 1, p=probs/np.sum(probs))[0]
                if selected in prevs:
                    continue
                prevs.add(selected)
                break
            self.arcs.extend([(len(self.deg), selected),(selected, len(self.deg))])            
            # self.weights=np.append(self.weights,[self.weights[-1]]*2)
            self.weights=np.ones(len(self.arcs))
            self.deg[selected] += 1
        self.deg = np.append(self.deg, m)
            
    def make_grover_matrix(self):
        return make_grover_matrix(self.deg,np.array(self.arcs))

@njit(parallel=True)
def make_grover_matrix(deg:NDArray,arcs:NDArray)->NDArray:
    n = len(arcs); mat = np.zeros((n,n))
    for i in prange(n):
        for j in prange(n):
            if arcs[i][1] == arcs[j][0]:                
                mat[j,i] = 2/(deg[arcs[i][1]]) - 1 if arcs[i][0] == arcs[j][1] else 2/(deg[arcs[i][1]])
    return mat
