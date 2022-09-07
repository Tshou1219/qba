import csv
from qba import QBA
from typing import List, Tuple


def write_data(qba: QBA, path: List[Tuple[int, float]]):
    dir = './test/'+qba.initial_graph+'/' + \
        str(len(qba.deg))+'_'+str(path)+'_'+str(qba.count[0])
    content = [qba.arcs, path, ['node', len(qba.deg)], ['count', qba.count[0]],
               qba.initial_state, qba.weights]
    with open(dir+'.csv', 'w') as f:
        writer = csv.writer(f)
        for content in content:
            writer.writerow(content)
