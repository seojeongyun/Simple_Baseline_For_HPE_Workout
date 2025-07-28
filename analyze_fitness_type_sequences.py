import os
import json

from glob import glob
from tqdm import tqdm
from collections import defaultdict

if __name__ == '__main__':
    # *-*-*-*-*-*- Collect json files from validation dataset *-*-*-*-*-*-
    base_path = '/storage/jysuh/fitness/fitness/validation/label'
    assert os.path.exists(base_path)

    json_list_valid = []        # 3139
    json_list_train = []        # 34468
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
                        json_list_train.append(day_path + '/' + json_files)
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    #
    #
    #
    #
    # *-*-*-*-*-*- extracting data from json files *-*-*-*-*-*-
    json_files_list = json_list_train + json_list_valid
    exercise_dict = {}
    counter = defaultdict(int)

    # -*-*-* Var for Debug *-*-*-
    check_sequences_dict = {}
    total = 0  # debug
    # -*-*-*-*-*-*-*-*-*-*-*-*-*-

    for json_file in tqdm(json_files_list, desc='Removing Missing Value Files...', leave=True):
        with open(json_file , 'r') as f:
            data = json.load(f)
            if len(data['frames']) == 0:
                json_files_list.remove(json_file)

            # if len(data['frames']) != 16:
                # if len(data['frames']) != 0:
                #     print("Debug")
                # elif len(data['frames']) == 0:
                #     json_files_list.remove(json_file)

    for json_file in tqdm(json_files_list, desc='Analyzing JSON Files...', leave=True):
        with open(json_file , 'r') as f:
            data = json.load(f)

            # translate '313-2-1-15-Z99_A-' -> '313-2-1-15-Z99_A'
            for frame_idx in range(len(data['frames'])):
                for view_idx in data['frames'][frame_idx].keys():
                    if '-/' in data['frames'][frame_idx][view_idx]['img_key']:
                        dir_name, file_name = os.path.split(data['frames'][frame_idx][view_idx]['img_key'])

                        dir_parts = dir_name.split('/')
                        dir_parts[-1] = dir_parts[-1][:-1]  # ??? ???? '-' ??
                        new_dir = '/'.join(dir_parts)

                        # ??? ?? ?? ???
                        new_path = os.path.join(new_dir, file_name)
                        data['frames'][frame_idx][view_idx]['img_key'] = new_path
            #
            parts = json_file.split('/')
            del parts[7]
            parts[6] = 'image'
            target_path = '/'.join(parts[:8])

            del data['type_info']['key']
            del data['type_info']['type']
            del data['type_info']['pose']

            # -*-*-* For Debug *-*-*-
            exercise = data['type_info']['exercise']
            check_sequences_dict.setdefault(exercise, {})
            check_sequences_dict[exercise].setdefault('sequences', [])
            check_sequences_dict[exercise]['sequences'].append(len(data['frames']))

            # -*-*-* Extracting data *-*-*-
            counter[exercise] += 1

            exercise_dict.setdefault(exercise, {})
            for num_view in range(1, 6):
                exercise_dict[exercise].setdefault(str(counter[exercise]), {})
                exercise_dict[exercise][str(counter[exercise])].setdefault(f'view{num_view}', {})
            exercise_dict[exercise][str(counter[exercise])].setdefault('type_info', data['type_info'])

            for num_view in range(1, 6):
                sequence_list = []
                for frame in range(len(data['frames'])):
                    # if data['frames'][frame][f'view{num_view}']['img_key'] == ''
                    sequence_list.append(os.path.join(target_path, data['frames'][frame][f'view{num_view}']['img_key']))
                exercise_dict[exercise][str(counter[exercise])][f'view{num_view}'].setdefault('img_path', sequence_list)

            # view1 --> 1 --> list(A sequence path)
            #           2 --> list(A sequence path)
            #           3 --> list(A sequence path)
            #               .
            #               .
            #               .
            #          end --> list(A sequence path)
            #               .
            #               .
            #               .
            # view5 --> end --> list(A sequence path)
    # Debug
    # for what_exer in exercise_dict.keys():
    #     for seq_num in exercise_dict[what_exer].keys():
    #         for view_num in exercise_dict[what_exer][seq_num].keys():
    #             if 'view' in view_num:
    #                 if len(exercise_dict[what_exer][seq_num][view_num]['img_path']) != 16:
    #                     print(
    #                         f"what_exer: {what_exer}, seq_num: {seq_num}, view_num: {view_num}, num_frames: {len(exercise_dict[what_exer][seq_num][view_num]['img_path'])}")

    with open('/storage/jysuh/Simple_Baseline_For_HPE_Workout/json_files/frame_sequences_w_type_info.json', 'w') as f:
        json.dump(exercise_dict, f)

    print(str(1))

    for k, v in check_sequences_dict.items():
        print(f"The number of sequences from {k}: {len(v['sequences'])}")
        total += len(v['sequences'])

    for k,v in exercise_dict.items():
        for num_view in range(1, 6):
            total += len(exercise_dict[k][f'view{num_view}'].key())

    A = 0

    for what_exer in exercise_dict.keys():
        for seq_num in exercise_dict[what_exer].keys():
            for view_num in exercise_dict[what_exer][seq_num].keys():
                if 'view' in view_num:
                    if len(exercise_dict[what_exer][seq_num][view_num]['img_path']) != 16:
                        print(f"what_exer: {what_exer}, seq_num: {seq_num}, view_num: {view_num}, num_frames: {len(exercise_dict[what_exer][seq_num][view_num]['img_path'])}")

    print(A)