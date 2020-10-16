import json
from general_config import config
from pathlib import Path,PurePath
import json


def create_dict():
    name_csv = config['raw_text_data_name']
    dir = PurePath(config['data_dir'])
    file_name = Path.joinpath(dir, name_csv)

    with open(file_name, 'r') as f:
        rows = [json.loads(row.split('\t')[0])['sentence1'] for row in f.readlines()[1:]]
    rows = list(set(rows))
    dict_to_save = {}
    for i, row in enumerate(rows):
        dict_to_save['sentence' + str(i)] = {'text': row}
    name_csv = config['text_data_name']
    file_name = Path.joinpath(dir, name_csv)
    with open(file_name, 'w') as fp:
        json.dump(dict_to_save, fp)


def load_text_data(limit_rows=False):
    name_csv = config['text_data_name']
    dir = PurePath(config['data_dir'])
    file_name = Path.joinpath(dir, name_csv)
    with open(file_name, 'r') as f:
        text_dict = json.load(f)

    if limit_rows:
        for i in range(limit_rows,len(text_dict.keys())):
            del text_dict['sentence'+str(i)]
    return text_dict
if __name__=='__main__':
    text_dict = load_text_data(limit_rows=100)
    print(len(text_dict.keys()))
    print(text_dict.keys())
