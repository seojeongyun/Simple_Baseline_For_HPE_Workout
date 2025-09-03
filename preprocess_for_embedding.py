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

    coord_dict = {}
    # ===
    for json_file in tqdm(json_files_list, desc='...', leave=True):
        with open(json_file , 'r') as f:
            data = json.load(f)
        #
        for frame_idx in range(len(data['frames'])):
            for view_idx in data['frames'][frame_idx].keys():
                coord_list = []
                head_x, head_y = 0, 0
                for joint_idx, joint_name in enumerate(data['frames'][frame_idx][view_idx]['pts'].keys()):
                    if joint_idx < 5:
                        head_x += data['frames'][frame_idx][view_idx]['pts'][joint_name]['x']
                        head_y += data['frames'][frame_idx][view_idx]['pts'][joint_name]['y']
                    elif joint_idx == 5:
                        head_x, head_y = head_x / 5, head_y / 5
                        coord_list.append([head_x, head_y])
                    else:
                        coord_list.append([data['frames'][frame_idx][view_idx]['pts'][joint_name]['x'], data['frames'][frame_idx][view_idx]['pts'][joint_name]['y']])



    #
    with open('/storage/jysuh/Simple_Baseline_For_HPE_Workout/json_files/exercise_conditions.json','w') as f:
        json.dump(exercise_dict, f)
