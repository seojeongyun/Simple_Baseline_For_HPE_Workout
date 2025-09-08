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
    coord_list = []
    # ===
    head_key = ['Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear']
    # for json_file in tqdm(json_files_list, desc='...', leave=True):
    for json_file in tqdm(json_list_valid, desc='...', leave=True):
        with open(json_file , 'r') as f:
            data = json.load(f)
            exercise_name = data['type_info']['exercise']
            coord_dict.setdefault(exercise_name, {})
        #
        for frame_idx in range(len(data['frames'])):
            coord_dict[exercise_name].setdefault(frame_idx, {})
            for view_idx in data['frames'][frame_idx].keys():
                coord_dict[exercise_name][frame_idx].setdefault(view_idx, [])
                coord_dict_for_a_frame = {}
                head_x, head_y = 0, 0
                for joint_idx, joint_name in enumerate(data['frames'][frame_idx][view_idx]['pts'].keys()):
                    if joint_name in head_key:
                        head_x += data['frames'][frame_idx][view_idx]['pts'][joint_name]['x']
                        head_y += data['frames'][frame_idx][view_idx]['pts'][joint_name]['y']
                    else:
                        coord_dict_for_a_frame[joint_name]= [data['frames'][frame_idx][view_idx]['pts'][joint_name]['x'], data['frames'][frame_idx][view_idx]['pts'][joint_name]['y']]
                head_x, head_y = head_x / 5, head_y / 5
                coord_dict_for_a_frame['head'] = [head_x, head_y]

                coord_dict[exercise_name][frame_idx][view_idx] = coord_dict_for_a_frame
    #
    with open('/storage/jysuh/Simple_Baseline_For_HPE_Workout/json_files/coord_data_valid.json', 'w') as f:
        json.dump(coord_dict, f)

# class make_embedding_vectors(nn.Module):
#     def __init__(self, vocab, data):
#         super(make_embedding_vectors, self).__init__()