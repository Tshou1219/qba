import csv
import os
import numpy as np
from qba import QBA
from networkx import DiGraph
from typing import List, Tuple
import matplotlib.pyplot as plt
import plot


def write_data(qba: QBA, path: List[Tuple[int, float]]):
    dir = './test/'+qba.initial_graph+'/' + str(len(qba.deg))+'_'+str(path)
    os.makedirs(dir)
    with open(dir+'/weight.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow([len(qba.deg)])
        writer.writerow(path)
        writer.writerow(qba.weights)
    plot.count_histgram(qba.count, True, dir+'/count.png')
    plot.deg_plot(DiGraph(qba.arcs), True, dir+'/deg.png')
    plot.plot_time(qba.times, True, dir+'/time.png')
