import csv
import json
from qba import QBA
from typing import List, Tuple


def write_data(qba: QBA, path: List[Tuple[int, float]]):
    dir = './test/'+qba.initial_graph+'/count_' + \
        str(len(qba.deg))+'_times_'+str(qba.count[0])
    content = [list(qba.arcs), path, [len(qba.deg)], [qba.count[0]],
               qba.initial_state, qba.weights]
    with open(dir+'.csv', 'w') as f:
        writer = csv.writer(f)
        for content in content:
            writer.writerow(content)


def write_json_data(qba: QBA, path: List[Tuple[int, float]]):

    print('********** JSONファイルを書き出す **********')

    # 辞書オブジェクト(dictionary)を生成
    data = dict()
    data['node'] = len(qba.deg)
    data['path'] = path
    data['count'] = qba.count
    data['weight'] = list(qba.weights)

    # 辞書オブジェクトをstr型で取得して出力
    print(json.dumps(data, ensure_ascii=False, indent=2))

    # 辞書オブジェクトをJSONファイルへ出力
    with open('mydata.json', mode='wt', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
