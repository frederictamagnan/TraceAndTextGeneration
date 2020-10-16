import numpy as np
from pathlib import Path,PurePath
from collections import defaultdict
from general_config import config
import json

def remove_brackets(data):
    for i,line in enumerate(data):
        for j,elt in enumerate(line):
            elt=elt.replace("[",'')
            elt=elt.replace(']','')
            line[j]=elt
        data[i]=line
    return data

def discretize_payer(data):
    for i,line in enumerate(data):
        if line[4]=="payer":
            if float(line[6])>0:
                line[5]='0'
                line[6]='0'
            else:
                line[5]='1'
                line[6]='0'
    return data


def create_raw_dict_by_sessions(data,columns_to_select=["object", "action", "parameter", "return_code"]):
    name_to_id_column = {"object": 3, "action": 4, "parameter": 5, "return_code": 6}

    def_dict = lambda:{"object":[] , "action":[] , "parameter": [], "return_code": []}
    dict_by_sessions = defaultdict(def_dict)
    clients = data[:, 2]

    for i,client_id in enumerate(clients):
        for column in columns_to_select:
            elt_to_append=data[i,name_to_id_column[column]]
            if elt_to_append=='':
                elt_to_append='None'
            dict_by_sessions[client_id][column].append(elt_to_append)
    return dict_by_sessions


def create_dict():
    name_csv = config['raw_traces_data_name']
    dir = PurePath(config['data_dir'])
    csv = np.loadtxt(Path.joinpath(dir, name_csv), delimiter=', ', dtype="object")
    data = remove_brackets(csv)
    data = discretize_payer(data)
    sessions = create_raw_dict_by_sessions(data)
    name_csv = config['traces_data_name']
    file_name = Path.joinpath(dir, name_csv)
    with open(file_name, 'w') as fp:
        json.dump(sessions, fp)

def load_traces_data(limit_rows=False):
    name_csv = config['traces_data_name']
    dir = PurePath(config['data_dir'])
    file_name = Path.joinpath(dir, name_csv)
    with open(file_name, 'r') as f:
        traces_dict = json.load(f)

    if limit_rows:
        for i in range(limit_rows,len(traces_dict.keys())):
            del traces_dict['client'+str(i)]
    return traces_dict

if __name__=='__main__':
    create_dict()
    traces_dict=load_traces_data(limit_rows=100)
    print(len(traces_dict.keys()))
    print(traces_dict.keys())
