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
        if 'json' not in os.listdir(base_path)[equipment_type_idx] and 'new' not in os.listdir(base_path)[equipment_type_idx]:
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

    # -*-*-* Var for Debug *-*-*-
    check_sequences_dict = {}
    total = 0  # debug
    # -*-*-*-*-*-*-*-*-*-*-*-*-*-

    # if "/storage/jysuh/fitness/fitness/train/label/body_Labeling/body_01/Day05_200925_F/D05-6-040.json" in json_files_list:
    #     print(1)
    # 2081
    # furniture 4262
    # body      15461
    # barbell   14745
    refined_json_files_list = []
    # =====================================================================================================================================
    # for json_file in tqdm(json_files_list, desc='Removing Missing Value Files...', leave=True):
    #     with open(json_file , 'r') as f:
    #         data = json.load(f)
    #         load_next_json_file = False
    #         # if json_file == "/storage/jysuh/fitness/fitness/train/label/body_Labeling/body_01/Day05_200925_F/D05-6-040.json":
    #         #     print(1)
    #         #
    #         parts = json_file.split('/')
    #         # parts: ['', 'storage', 'jysuh', 'fitness', 'fitness', 'train', 'image', 'barbell_dumbbell_Labeling', 'babel_13', 'Day42_201116_F', 'D42-7-682.json']
    #         #
    #         del parts[7]
    #         # parts[7] = 'barbell_dumbbell_Labeling'
    #         #
    #         parts[6] = 'image'
    #         target_path = '/'.join(parts[:8])
    #         # target_path = '/storage/jysuh/fitness/fitness/train/image/babel_13'
    #         #
    #
    #         if len(data['frames']) == 0:
    #             continue
    #         #
    #         # Are there all img_paths ?
    #         for view_idx in range(1, 6):
    #             for frame_idx in range(len(data['frames'])):
    #                 if json_file in json_files_list:
    #                     img_path = os.path.join(target_path, data['frames'][frame_idx][f'view{view_idx}']['img_key'])
    #                     if '-/' in data['frames'][frame_idx][f'view{view_idx}']['img_key']:
    #                         # translate '313-2-1-15-Z99_A-' -> '313-2-1-15-Z99_A'
    #                         dir_name, file_name = os.path.split(data['frames'][frame_idx][f'view{view_idx}']['img_key'])
    #
    #                         dir_parts = dir_name.split('/')
    #                         dir_parts[-1] = dir_parts[-1][:-1]
    #                         new_dir = '/'.join(dir_parts)
    #
    #                         new_path = os.path.join(new_dir, file_name)
    #                         new_path = os.path.join(target_path, new_path)
    #                         if not os.path.exists(new_path):
    #                             load_next_json_file = True
    #                             break
    #
    #                     elif not os.path.exists(img_path):
    #                         load_next_json_file = True
    #                         break
    #
    #             if load_next_json_file:
    #                 break
    #
    #     if not load_next_json_file:
    #         refined_json_files_list.append(json_file)
    #
    # with open('/storage/jysuh/Simple_Baseline_For_HPE_Workout/json_files/frame_sequences_w_type_info_debug.json','w') as f:
    #     json.dump(refined_json_files_list, f)
    # =====================================================================================================================================

            # if len(data['frames']) != 16:
                # if len(data['frames']) != 0:
                #     print("Debug")
                # elif len(data['frames']) == 0:
                #     json_files_list.remove(json_file)
    with open('/json_files/frame_sequences_w_type_info_debug.json', 'r') as f:
        json_files_list = json.load(f)

    max_frame = 0
    for json_file in tqdm(json_files_list, desc='Analyzing JSON Files...', leave=True):
        with open(json_file , 'r') as f:
            data = json.load(f)

            if max_frame < len(data['frames']):
                max_frame = len(data['frames'])

            # convert '313-2-1-15-Z99_A-' -> '313-2-1-15-Z99_A'
            for frame_idx in range(len(data['frames'])):
                for view_idx in data['frames'][frame_idx].keys():
                    if '-/' in data['frames'][frame_idx][view_idx]['img_key']:
                        dir_name, file_name = os.path.split(data['frames'][frame_idx][view_idx]['img_key'])

                        dir_parts = dir_name.split('/')
                        dir_parts[-1] = dir_parts[-1][:-1]
                        new_dir = '/'.join(dir_parts)

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
                    if 'Day05_200925_F/6/A/040-1-1-02-Z22_A/040-1-1-02-Z22_A-0000001.jpg' == data['frames'][frame][f'view{num_view}']['img_key']:
                        print(1)
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
    #
    exercise_dict['max_frame'] = max_frame
    #
    with open('/json_files/frame_sequences_w_type_info.json', 'w') as f:
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