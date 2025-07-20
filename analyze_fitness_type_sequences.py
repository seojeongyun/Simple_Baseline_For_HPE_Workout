import os
import json
import torch
import glob
import numpy as np

from glob import glob
import fnmatch
from tqdm import tqdm

if __name__ == '__main__':
    # *-*-*-*-*-*- Collect json files from validation dataset *-*-*-*-*-*-
    base_path = '/storage/jysuh/fitness/fitness/validation/label'
    assert os.path.exists(base_path)

    json_list_valid = []        # 3139
    json_list_train = []        # 34468

    for equipment_type_idx, _ in enumerate(os.listdir(base_path)):
        path = base_path + '/' + os.listdir(base_path)[equipment_type_idx]
        for idx in range(len(os.listdir(path))):
            path = path + '/' + os.listdir(path)[idx]
            for idx in range(len(os.listdir(path))):
                path = path + '/' + os.listdir(path)[idx]
        for _, json_files in enumerate(os.listdir(path)):
            if '3d' not in json_files:
                json_list_valid.append(path + json_files)
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    #
    #
    #
    # *-*-*-*-*-*- Collect json files from train dataset *-*-*-*-*-*-
    base_path = '/storage/jysuh/fitness/fitness/train/label'

    for equipment_type_idx, _ in enumerate(os.listdir(base_path)):
        if 'json' not in os.listdir(base_path)[equipment_type_idx] and 'new' not in os.listdir(base_path)[equipment_type_idx]:
            equipment_type_path = base_path + '/' + os.listdir(base_path)[equipment_type_idx]
            for dir_idx in range(len(os.listdir(equipment_type_path))):
                dir_path = equipment_type_path + '/' + os.listdir(equipment_type_path)[dir_idx]
                for idx in range(len(os.listdir(dir_path))):
                    day_path = dir_path + '/' + os.listdir(dir_path)[idx]

                for _, json_files in enumerate(os.listdir(day_path)):
                    if '3d' not in json_files:
                        json_list_train.append(day_path + json_files)
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    print(1)