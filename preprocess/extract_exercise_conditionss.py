import os
import json

from glob import glob
import pickle
from tqdm import tqdm
from collections import defaultdict
from collections import Counter

if __name__ == '__main__':
    # *-*-*-*-*-*- Collect json files from validation dataset *-*-*-*-*-*-
    base_path = '/storage/jysuh/fitness/fitness/validation/label'
    assert os.path.exists(base_path)

    json_list_valid = []  # 3139
    json_list_train = []  # 34468
    # total number of json files is 37670

    for equipment_type_idx, _ in enumerate(os.listdir(base_path)):
        path = base_path + '/' + os.listdir(base_path)[equipment_type_idx]
        for idx in range(len(os.listdir(path))):
            path = path + '/' + os.listdir(path)[idx]
            for idx in range(len(os.listdir(path))):
                path = path + '/' + os.listdir(path)[idx]
        for _, json_files in enumerate(os.listdir(path)):
            if '3d' not in json_files:
                json_list_valid.append(path + '/' + json_files)
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    #

    base_path = '/storage/jysuh/fitness/fitness/train/label'

    for equipment_type_idx, _ in enumerate(os.listdir(base_path)):
        if 'json' not in os.listdir(base_path)[equipment_type_idx] and 'new' not in os.listdir(base_path)[
            equipment_type_idx]:
            equipment_type_path = '/'.join([base_path, os.listdir(base_path)[equipment_type_idx]])
            for equipment_idx in range(len(os.listdir(equipment_type_path))):
                equipment_path = '/'.join([equipment_type_path, os.listdir(equipment_type_path)[equipment_idx]])
                for idx in range(len(os.listdir(equipment_path))):
                    day_path = '/'.join([equipment_path, os.listdir(equipment_path)[idx]])

                for _, json_files in enumerate(os.listdir(day_path)):
                    if '3d' not in json_files:
                        json_list_train.append(day_path + '/' + json_files)
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-



    # *-*-*-*-*-*- extracting data from json files *-*-*-*-*-*-
    json_files_list = json_list_train + json_list_valid
    condition_vocab = {}
    debug = []
    a_debug = []
    b_debug = []
    val = 63 # the last number in train_vocab is 62. thus, condition start from 63
    for json_file in tqdm(json_list_train, desc='...', leave=True):
        with open(json_file , 'r') as f:
            data = json.load(f)
            exercise_name = data['type_info']['exercise']
            if exercise_name == '덤벨 인클라인 체스트 플라이' and len(a_debug) == 0:
                a_debug.append(data['type_info']['conditions'])
            elif exercise_name == '덤벨 체스트 플라이' and len(b_debug) == 0:
                b_debug.append(data['type_info']['conditions'])

            for i in range(len(data['type_info']['conditions'])):
                condition = data['type_info']['conditions'][i]['condition']
                if condition not in debug:
                    debug.append(condition)
                condition_vocab.setdefault(condition, 0)
                if condition_vocab[condition] == 0:
                    condition_vocab[condition] = val
                    val += 1

    with open('/home/jysuh/PycharmProjects/coord_embedding/dataset/embedder_dataset/condition_vocab.pkl', 'wb') as f:
        pickle.dump(condition_vocab, f)