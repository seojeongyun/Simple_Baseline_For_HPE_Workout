import os
import json

from glob import glob
from tqdm import tqdm
from collections import defaultdict

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
    #
    # ========= For Debug ==========
    # equipment_type_list = []
    # bd_list = []
    # furniture_list = []
    # body_list = []
    # body_idx_dict = {}
    # counter = defaultdict(int)
    # *-*-*-*-*-*- Collect json files from train dataset *-*-*-*-*-*-
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
    # for k, v in counter.items():
    #     print(f'{k} : {v}')
    #
    #
    #
    #
    # *-*-*-*-*-*- extracting data from json files *-*-*-*-*-*-
    json_files_list = json_list_train + json_list_valid
    exercise_dict = {}
    counter = defaultdict(int)

    # ===
    for json_file in tqdm(json_files_list, desc='...', leave=True):
        with open(json_file , 'r') as f:
            data = json.load(f)
            exercise = data['type_info']['exercise']
            exercise_dict.setdefault(exercise, {'condition':[], 'First Time':True})
            for condition_idx in range(len(data['type_info']['conditions'])):
                if data['type_info']['conditions'][condition_idx]['condition'] not in exercise_dict[exercise]['condition'] and exercise_dict[exercise]['First Time']:
                    exercise_dict[exercise]['condition'].append(data['type_info']['conditions'][condition_idx]['condition'])
                elif data['type_info']['conditions'][condition_idx]['condition'] not in exercise_dict[exercise]['condition'] and exercise_dict[exercise]['First Time'] == False:
                    raise ValueError('Stop')
                else:
                    continue
            exercise_dict[exercise]['First Time'] = False

    # with open('/storage/jysuh/Simple_Baseline_For_HPE_Workout/json_files/exercise_conditions.json', 'r') as f:
    #     exercise_dict = json.load(f)
    #
    #
    # JOINTS_DICT = {
    # "눈": (1, 2),
    # "귀": (3, 4),
    # "어깨": (5, 6),
    # "팔꿈치": (7, 8),
    # "손목": (9, 10),
    # "엉덩이": (11, 12),
    # "무릎": (13, 14),
    # "발목": (15, 16),
    # "목": 17,
    # "손바닥": (18, 19),
    # "등": 20,
    # "허리": 21,
    # "발": (22, 23),
    # }
    #
    # for exer_name in exercise_dict.keys():
    #     for condition_idx in range(len(exercise_dict[exer_name]['condition'])):
    #         condition_split = exercise_dict[exer_name]['condition'][condition_idx].split()
    #         for joint_name in JOINTS_DICT.keys():
    #             if joint_name in condition_split:
    #                 exercise_dict[exer_name]['condition'][condition_idx] = {exercise_dict[exer_name]['condition'][condition_idx], joint_name}


    with open('/storage/jysuh/Simple_Baseline_For_HPE_Workout/json_files/exercise_conditions.json','w') as f:
        json.dump(exercise_dict, f)
