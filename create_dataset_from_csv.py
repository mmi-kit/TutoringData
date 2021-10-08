#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np
import glob
import json
import os
from collections import OrderedDict as dic

annotation_dir = "./annotation_data"
annotation1_path = os.path.join(annotation_dir, "annotation1")
annotation2_path = os.path.join(annotation_dir, "annotation2")
annotation3_path = os.path.join(annotation_dir, "annotation3")

save_path = "./zyouhou1_hatsumon.json"


# In[2]:


def load_data(annotation_dir: str) -> np.ndarray:
    
    datas = []
    files = glob.glob(annotation_dir + "/*") #各アノテーションファイルpair*-*を取得

    for file in files:
        data = get_data_from_file(file)
        datas.extend(data)
        
    np_datas = np.empty((len(datas), len(datas[0])), dtype = object)
    np_datas[:] = datas
        
    return np_datas   # datas[i] = [question_number, question, model_ans, question_type, edu_ans, label, , , ]


# In[3]:


def get_data_from_file(file_path: str) -> list:
    with open(file_path, "r") as f:
        pair = csv.reader(f)
        data = [p for p in pair]
        data = data[2:]       #最初の2行は省く
        
        return data


# In[4]:


def str_label_to_int(str_label: str) -> int:
    int_label = int(str_label.split(":")[1].replace(" ", ""))
    return int_label


# In[5]:


def add_one_to_label_match_matrix(i_label_match_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = labels - 1     # label: 1~7, index : 0~6
    labels = labels.tolist()
    
    for l in labels:
        i_label_match_matrix[l] += 1
    return i_label_match_matrix


# In[6]:


def label_match_matrix(annotation_datas: np.ndarray) -> np.ndarray:
    
    # annotation_data = [num_annotator, num_data, ]
    
    n_datas = annotation_datas.shape[1]  #n_datas = 450
    n_labels = 7
    label_match_matrix = np.zeros((n_datas, n_labels), dtype = int)
    
    labels = annotation_datas[:, :, 5].T   # shape[n_datas, num_annotator]
    convert_str_to_int = np.frompyfunc(str_label_to_int, 1, 1)
    labels = convert_str_to_int(labels)
    
    for i in range(label_match_matrix.shape[0]):
        label_match_matrix[i] = add_one_to_label_match_matrix(label_match_matrix[i], labels[i])
    
    return label_match_matrix


# In[7]:


def compute_fleiss_kappa(label_match_matrix: np.ndarray, num_annotator: int) -> None:
    
    
    label_match_matrix = label_match_matrix.tolist()
    
    N = len(label_match_matrix)
    k = len(label_match_matrix[0])
    # 入力された情報の確認
    print('評価者の数 = {}'.format(num_annotator))
    print('評価対象の数 = {}'.format(N))
    print('評価カテゴリー数 = {}'.format(k))

    # Piの値を求めて，P_barを求める
    P_bar = sum([(sum([el**2 for el in row]) - num_annotator) / (num_annotator * (num_annotator - 1)) for row in label_match_matrix]) / N
    print('P_bar  = {}'.format(P_bar))

    # pjの値を求めて，Pe_barを求める
    Pe_bar = sum([(sum([row[j] for row in label_match_matrix]) / (N * num_annotator)) ** 2 for j in range(k)])
    print('Pe_bar  = {}'.format(Pe_bar))

    # fleiss kappa値の計算
    kappa = float(0)
    try:
        kappa = (P_bar - Pe_bar) / (1 - Pe_bar)
    except ZeroDivisionError:
        kappa = float(1)

    print("kappa = {}".format(kappa))
    
    if kappa < 0:
        print("一致していない")
    elif kappa <= 0.20:
        print("わずかに一致")
    elif kappa <= 0.40:
        print("だいたい一致")
    elif kappa <= 0.60:
        print("適度に一致")
    elif kappa <= 0.80:
        print("かなり一致")
    elif kappa <= 1.00:
        print("ほとんど一致")


# In[40]:


def label_majority_vote(hatsumon_pair: np.ndarray, label_match_matrix: np.ndarray) -> np.ndarray:
    # hatsumon_pair[i] = [question_number, question, model_ans, edu_ans]
    
    majority_vote = np.argmax(label_match_matrix, axis=1)
    majority_vote = majority_vote.reshape(majority_vote.shape[0], 1) + 1 # index: 0~6 label: 1~7
    
    hatsumon_pair = np.hstack((hatsumon_pair, majority_vote))
    
    return hatsumon_pair


# In[46]:


def create_dataset(np_hatsumon_pair: np.ndarray)  -> dict:
    chap = 1
    gakusyuu = 1
    num_hatsumon = 1

    dataset = dic()
    
    for g in range(gakusyuu, 25):
        gakusyu_index = "Study" + str(g)
        if g == 6:
            chap += 1
        elif g == 11:
            chap += 1
        elif g == 18:
            chap += 1
    
        dataset[gakusyu_index] = dic()
        
        for h in range(num_hatsumon, 6):
            index = str(chap)+"-"+str(g)+"-"+str(h)
            data = [d for d in np_hatsumon_pair if d[0] == index]
            
            hatsumon_pair = dic()
            hatsumon_pair["question"] = data[0][1]
            hatsumon_pair["model_ans"] = data[0][2]
            edu_anss = [{"ans": d[3], "label": d[4]} for d in data]
            hatsumon_pair["edu_ans"] = edu_anss
            q_index = "Question" + str(h)
            dataset[gakusyu_index][q_index] = hatsumon_pair
            
    return dataset


# In[ ]:


def write_json(dataset: dict, save_path: str):
    with open(save_path, "x") as f:
        json.dump(dataset, f)


# In[ ]:


def main():
    
    if os.path.exists(save_path):
        print("Already exists zyouhou1 json data")
    else:
        
        print(" Create json data from dir = {}".format(annotation_dir))

        anotation1 = load_data(annotation1_path)
        anotation2 = load_data(annotation2_path)
        anotation3 = load_data(annotation3_path)

        annotation_datas = np.empty((3,anotation1.shape[0], anotation1.shape[1]), dtype = object)
        annotation_datas[0]  = anotation1
        annotation_datas[1]  = anotation2
        annotation_datas[2]  = anotation3

        hatsumon_pair = np.empty((annotation_datas.shape[1], 4), dtype = object)
        hatsumon_pair[:, 0] = anotation1[:, 0]
        hatsumon_pair[:, 1] = anotation1[:, 1]
        hatsumon_pair[:, 2] = anotation1[:, 2]
        hatsumon_pair[:, 3] = anotation1[:, 4]

        label_matrix = label_match_matrix(annotation_datas)
        compute_fleiss_kappa(label_matrix, num_annotator=3)

        np_dataset = label_majority_vote(hatsumon_pair, label_matrix)
        dataset = create_dataset(np_dataset)

        write_json(dataset, save_path)

        print("Complete")


# In[ ]:


if __name__ == "__main__":
    main()

