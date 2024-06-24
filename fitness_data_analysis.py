import os
import json
import torch
import glob
import numpy as np

from glob import glob
from tqdm import tqdm

if __name__ == '__main__':
    import glob

    base_path = '/storage/jysuh/fitness/fitness/train/label'

    assert os.path.exists(base_path)

    # ===== get_json_files =====
    label_path = []
    json_list = []

    for path in os.listdir(base_path):
        if 'json' not in path: # and 'furniture' in path or 'new' in path
            label_path.append(base_path + '/' + path)

    for fitness_type_idx, _ in enumerate(tqdm(label_path, desc="collect every json file in workout directory  ")):
        for _, num in enumerate(os.listdir(label_path[fitness_type_idx])):
            path = os.path.join(label_path[fitness_type_idx] + '/' + num)
            path = path + '/' + os.listdir(path)[0]
            json_files = glob.glob(path + '/' + '*.json')
            for _, json_file in enumerate(json_files):
                if '3d' not in (json_file):
                    json_list.append(json_file) # 34468
    # ===== ===== ===== ===== =====



    # ===== get_data =====
    # data_dict = {'pts' : [] ,
    #              'img_path' : [],
    #              'joints' : [],
    #              'joints_vis' : []}
    data_dict = {}

    # cnt = 0
    # jnt = 0
    # for json_idx in range(len(json_list)):
    #     with open(json_list[json_idx], 'r') as f:
    #         data = json.load(f)
    #         for idx in range(len(data['frames'])):
    #             for j, view in enumerate(data['frames'][idx]):
    #                 cnt += 1
    #
    # print(cnt)


    for json_idx in tqdm(range(len(json_list)), leave=True, desc="extracting img path and pts from json file  "):
        json_file = json_list[json_idx]
        exercise_path = json_list[json_idx].replace('label','image').split('/')
        if 'barbell_dumbbell' in exercise_path[8]:
            num = exercise_path[8].split('_')[2]
            exercise_path[8] = 'babel'
            exercise_path[8] = '_'.join([exercise_path[8], num])
        exercise_path = os.path.join('/'.join(exercise_path[0:7]), '/'.join(exercise_path[8:-2]))
        with open(json_file , 'r') as f:
            data = json.load(f)
            for frame_idx in range(len(data['frames'])):
                for idx, view_idx in enumerate (data['frames'][frame_idx].keys()):
                    # PTS
                    img_path = os.path.join(exercise_path, data['frames'][frame_idx][view_idx]['img_key'])
                    # cnt += 1
                    if img_path not in data_dict.keys():
                        # jnt += 1
                        # data_dict[img_path]['pts'] = data['frames'][frame_idx][view_idx]['pts']
                        # data_dict['pts'].append(data['frames'][frame_idx][view_idx]['pts'])
                        # data_dict['img_path'].append(os.path.join(exercise_path, data['frames'][frame_idx][view_idx]['img_key']))

                        # JOINTS
                        joints = []
                        joints_vis = []
                        #
                        data_dict.setdefault(img_path, {'joints': None})
                        data_dict.setdefault(img_path, {'joints_vis': None})
                        # data_dict[img_path].setdefault('joints', None)
                        # data_dict[img_path].setdefault('joints_vis', None)
                        #
                        for joint_idx, joint in enumerate(data['frames'][frame_idx][view_idx]['pts'].keys()):
                            joint_pts = [data['frames'][frame_idx][view_idx]['pts'][joint]['x'], data['frames'][frame_idx][view_idx]['pts'][joint]['y']]
                            joints.append(joint_pts)

                            joint_visibility = [1, 1]
                            joints_vis.append(joint_visibility)

                        data_dict[img_path]['joints'] = joints
                        data_dict[img_path]['joints_vis'] = joints_vis
                        # del data_dict[img_path]['pts']

    with open('/storage/jysuh/Simple_Baseline_For_HPE_Workout/data_pts_del.json', 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)
    # ===== ===== ===== =====

    # print(sorted(json_list))    # the number of 2d json files are 34468

    fitness_dict = {}
    exer = dict(
        condition=[],
        description=[],
    )
    #
    description_anal = {}
    exer_type = []

    cnt = 0

    exer_name_list = []

    for idx in range(len(json_list)):
        with open(json_list[idx]) as f:
            exer_name = json.load(f)['type_info']['exercise']
            if exer_name not in exer_name_list:
                exer_name_list.append(exer_name)

    for idx in range(len(exer_name_list)):
        exer_type.append([])

    for idx in range(len(json_list)):
        with open(json_list[idx]) as f:
            annot = json.load(f)
            # ==== extract data ====
            exer_name = annot['type_info']['exercise']
            #
            attrs = annot['type_info']['conditions']
            #
            description = annot['type_info']['description']
            #
            data = []
            #
            # ==== make dict ====
            if not exer_name in fitness_dict.keys():
                fitness_dict.setdefault(exer_name, dict(condition=[], description=[]))
            #
            if not exer_name in description_anal.keys():
                description_anal.setdefault(exer_name, [])

            for idx in range(len(attrs)):
                data.append(attrs[idx]['value'])
                if idx == len(attrs)-1:
                    data.append(description)

            for idx, name in enumerate(exer_name_list):
                if name == exer_name:
                    exer_type[idx].append(data)
                    break

            if exer_name not in exer_name_list:
                cnt += 1
                print(exer_name)


            if len(fitness_dict[str(exer_name)]['condition']) == 0:
                for idx in range(len(attrs)):
                    fitness_dict[str(exer_name)]['condition'].append(attrs[idx]['condition'])

            if not description in fitness_dict[str(exer_name)]['description']:
                    fitness_dict[str(exer_name)]['description'].append(description)

    for i, name in enumerate(exer_name_list):
        for j, name_ in enumerate(description_anal.keys()):
            if name == name_:
                description_anal[str(name_)].append(exer_type[i])
                break

    # print(fitness_dict)

description_exer_desc = dict()
for _, exer_name in enumerate(tqdm(description_anal.keys(), desc="make json file for feedback model")):
    description_exer_desc.setdefault(exer_name, dict())
    for idx in range(len(description_anal[exer_name][0])):
        idx_bool = [np.where(np.array([isinstance(description_anal[exer_name][0][idx][bool_idx], bool) for bool_idx in range(len(description_anal[exer_name][0][idx]))]))[0]][0]
        value_only = [description_anal[exer_name][0][idx][b_idx] for b_idx in idx_bool]
        key_value = np.sum([(2**((len(value_only)-1) - aa)) * bool(value_only[aa]) for aa in range(len(value_only))])
        #
        idx_all = list(range(len(description_anal[exer_name][0][idx])))
        for re_value in list(idx_bool):
            idx_all.remove(re_value)
        #
        if key_value not in description_exer_desc[exer_name].keys():
            description_exer_desc[exer_name].setdefault(int(key_value), [])
            for idx_str in idx_all:
                description_exer_desc[exer_name][key_value].append((value_only, description_anal[exer_name][0][idx][idx_str]))

with open('/storage/jysuh/Simple_Baseline_For_HPE_Workout/feedback_model_label.json', 'w', encoding='utf-8') as f:
    json.dump(description_exer_desc, f, ensure_ascii=False, indent=4)


