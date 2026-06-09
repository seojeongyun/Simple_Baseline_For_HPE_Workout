import os
import json

from glob import glob
import pickle
from tqdm import tqdm
from collections import defaultdict
from collections import Counter


if __name__ == '__main__':
    # *-*-*-*-*-*- Collect validation json files *-*-*-*-*-*-
    base_path = '/storage/jysuh/fitness/fitness/validation/label'
    assert os.path.exists(base_path)

    json_list_valid = []  # 3139
    json_list_train = []  # 34468
    # total number of json files is 37670

    # [Validation] max_frame = 17
    # [Train] max_frame = 21

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

    # *-*-*-*-*-*- Collect train json files *-*-*-*-*-*-
    base_path = '/storage/jysuh/fitness/fitness/train/label'

    for equipment_type_idx, _ in enumerate(os.listdir(base_path)):
        if 'json' not in os.listdir(base_path)[equipment_type_idx] and 'new' not in os.listdir(base_path)[
            equipment_type_idx]:
            # equipment_type_list.append(os.listdir(base_path)[equipment_type_idx]) # DB
            equipment_type_path = '/'.join([base_path, os.listdir(base_path)[equipment_type_idx]])
            for equipment_idx in range(len(os.listdir(equipment_type_path))):
                equipment_path = '/'.join([equipment_type_path, os.listdir(equipment_type_path)[equipment_idx]])
                for idx in range(len(os.listdir(equipment_path))):
                    day_path = '/'.join([equipment_path, os.listdir(equipment_path)[idx]])

                for _, json_files in enumerate(os.listdir(day_path)):
                    if '3d' not in json_files:
                        # if os.listdir(base_path)[equipment_type_idx] == 'barbell_dumbbell_Labeling':    # DB
                        #     bd_list.append(day_path + '/' + json_files) # DB
                        # elif os.listdir(base_path)[equipment_type_idx] == 'furniture_Labeling': # DB
                        #     furniture_list.append(day_path + '/' + json_files) # DB
                        # elif os.listdir(base_path)[equipment_type_idx] == 'body_Labeling': # DB
                        #     counter[day_path.split('/')[8]] += 1
                        #     body_list.append(day_path + '/' + json_files)   # DB
                        json_list_train.append(day_path + '/' + json_files)
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

    # *-*-*-*-*-*- extracting information from json files *-*-*-*-*-*-
    json_files_list = json_list_train + json_list_valid

    with open("/storage/jysuh/Simple_Baseline_For_HPE_Workout/demo/valid_json_files_path.json", "w", encoding="utf-8") as f:
        json.dump(json_list_valid, f, ensure_ascii=False)
    print()
