#!/usr/bin/env python
# coding: utf-8

# In[83]:


import random
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
import json
import os
import pandas as pd


# In[84]:


class HatsumonDataGenerator:
    def __init__(self, path: str, tokenizer):
        self.two_way = {"1": 1, "2": 1, "3": 1, "4": 1, "5": 2, "6": 2, "7": 2}
        self.four_way = {"1": 1, "2": 1, "3": 2, "4": 2, "5": 3, "6": 3, "7": 4}
        self.seven_way = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
        self.bert_tokenizer = tokenizer
        self.json_path = path
        self.dataset = self.load_json_file(self.json_path)
        self.num_cross_val = 5
        
    def load_json_file(self, path: str) -> dict:
        with open(path, "r") as j:
            dataset = json.load(j)
        return dataset
    
    def view_all_data(self, way_type: str) -> pd.DataFrame:
        TUA_idx = np.zeros((self.num_cross_val, 1), dtype = str)
        TUQ_idx = [[]] * self.num_cross_val   
        
        question_numbers = []
        questions = []
        model_anss = []
        edu_anss = []
        labels = []
        
        for study_number in self.dataset.keys():
            for q, question_number in enumerate(self.dataset[study_number].keys()):
                edu_num = len(self.dataset[study_number][question_number]["edu_ans"])
                q_number = study_number + "-" + question_number
                question = self.dataset[study_number][question_number]["question"]
                model_ans = self.dataset[study_number][question_number]["model_ans"]

                q_number_list = [q_number] * edu_num
                questions.extend([question] * edu_num)
                model_anss.extend([model_ans] * edu_num)
                
                for i, edu_ans in enumerate(self.dataset[study_number][question_number]["edu_ans"]):
                    ans = edu_ans["ans"]
                    label = edu_ans["label"]
                    q_number_list[i] += "-" + str(i)
                    edu_anss.append(ans)
                    labels.append(str(label))
                    
                TUA_idx = self.create_TUA_idx(TUA_idx, q_number_list)
                TUQ_idx = self.create_TUQ_idx(TUQ_idx, q_number_list, q)
                question_numbers.extend(q_number_list)
                
                    
        pd_question_numbers = pd.DataFrame(np.array(question_numbers, dtype = str), columns=["Question number"])
        pd_questions = pd.DataFrame(np.array(questions, dtype = str), columns=["Question"])
        pd_model_anss = pd.DataFrame(np.array(model_anss, dtype = str), columns=["Model answer"])
        pd_edu_anss = pd.DataFrame(np.array(edu_anss, dtype = str), columns=["Edu answer"])
        labels = self.replace_way_type(labels, way_type)
        pd_labels = pd.DataFrame(np.array(labels, dtype = str), columns=["Label"])
        
        pd_all_data = pd.concat([pd_question_numbers, pd_questions, pd_model_anss, pd_edu_anss, pd_labels], axis=1)
        
        return pd_all_data, np.delete(TUA_idx, 0, 1).tolist(), TUQ_idx
    
    def create_TUA_idx(self, TUA_idx:np.ndarray, q_numbers:list):
        
        if len(q_numbers) < self.num_cross_val:
            ignores = ["ignore"] * (self.num_cross_val - len(q_numbers))
            q_numbers = q_numbers + ignores
        
        np_q_numbers = np.array(q_numbers, dtype=str).T.reshape(self.num_cross_val,1)
        
        TUA_idx = np.hstack((TUA_idx, np_q_numbers))
            
        return TUA_idx
    
    def create_TUQ_idx(self, TUQ_idx:list, q_numbers:list, q):
        idx = TUQ_idx[q] + q_numbers
        TUQ_idx[q] = idx
        return TUQ_idx
        
        
                
                
            
    def get_TUA(self, way_type: str) -> list:
        num_cross_val = self.num_cross_val
        train_pairs = np.zeros((num_cross_val, 1), dtype = str)
        train_labels = np.zeros((num_cross_val, 1), dtype = str)
        test_pairs = np.zeros((num_cross_val, 1), dtype = str)
        test_labels = np.zeros((num_cross_val, 1), dtype = str)
        
        for study_number in self.dataset.keys():
            for question_number in self.dataset[study_number].keys():
                question = self.dataset[study_number][question_number]["question"]
                model_ans = self.dataset[study_number][question_number]["model_ans"]
                
                pairs = []
                labels = []
                for edu_ans in self.dataset[study_number][question_number]["edu_ans"]:
                    ans = edu_ans["ans"]
                    pair = question + self.bert_tokenizer.tokenizer.sep_token + model_ans + self.bert_tokenizer.tokenizer.sep_token + ans
                    label = edu_ans["label"]
                    pairs.append(pair)
                    labels.append(label)
                    
                pairs = [pairs] * num_cross_val  # pairs = [num_cross_val, len(edu_ans)]
                labels = [labels] * num_cross_val # labelss = [num_cross_val, len(labels)]
                splitted_pairs = np.array(list(map(self.split_train_test_for_TUA, list(enumerate(pairs)))), dtype = str)
                splitted_labels = np.array(list(map(self.split_train_test_for_TUA, list(enumerate(labels)))), dtype = str)
                train_pairs = np.hstack((train_pairs, splitted_pairs[:, :-1])) # dim=1にどんどんスタックしていく
                train_labels = np.hstack((train_labels, splitted_labels[:, :-1]))
                test_pairs = np.hstack((test_pairs, splitted_pairs[:, -1].reshape(num_cross_val, 1)))
                test_labels = np.hstack((test_labels, splitted_labels[:, -1].reshape(num_cross_val, 1)))
                
        train_dataset = self.encording(train_pairs, train_labels, way_type)
        test_dataset = self.encording(test_pairs, test_labels, way_type)
        
        return train_dataset, test_dataset
                
            
        
    def get_TUQ(self, way_type: str) -> list:
        num_cross_val = self.num_cross_val
        train_pairs = np.zeros((num_cross_val, 1), dtype = str)
        train_labels = np.zeros((num_cross_val, 1), dtype = str)
        test_pairs = np.zeros((num_cross_val, 1), dtype = str)
        test_labels = np.zeros((num_cross_val, 1), dtype = str)
        
        for study_number in self.dataset.keys():
            study_pairs = [] #study_pairs[i] = Question_i
            study_labels = [] #study_labels[i] = Question_i
            for question_number in self.dataset[study_number].keys():
                question = self.dataset[study_number][question_number]["question"]
                model_ans = self.dataset[study_number][question_number]["model_ans"]
                
                pairs = []
                labels = []
                for edu_ans in self.dataset[study_number][question_number]["edu_ans"]:
                    ans = edu_ans["ans"]
                    pair = question + self.bert_tokenizer.tokenizer.sep_token + model_ans + self.bert_tokenizer.tokenizer.sep_token + ans
                    label = edu_ans["label"]
                    pairs.append(pair)
                    labels.append(label)
                    
                study_pairs.append(pairs)
                study_labels.append(labels)
                
            study_pairs = [study_pairs] * num_cross_val
            study_labels = [study_labels] * num_cross_val
            
            np_train_pairs, np_train_labels, np_test_pairs, np_test_labels = self.split_train_test_for_TUQ(study_pairs, study_labels)
            train_pairs = np.hstack((train_pairs, np_train_pairs))
            train_labels = np.hstack((train_labels, np_train_labels))
            test_pairs = np.hstack((test_pairs, np_test_pairs))
            test_labels = np.hstack((test_labels, np_test_labels))                
        
        train_dataset = self.encording(train_pairs, train_labels, way_type)
        test_dataset = self.encording(test_pairs, test_labels, way_type)
        
        return train_dataset, test_dataset
    
    
    def split_train_test_for_TUA(self, data_add_index: list) -> list:
        index = data_add_index[0]
        data = data_add_index[1]
        
        if len(data) <= index:
            random.shuffle(data)
            testset = data[-1]
            trainset = data[:-1]
        else:
            testset = data[index]
            trainset = data[:index] + data[index+1:]
        
        data = trainset + [testset]
            
        return data  # data[:-1] = train_data,  data[-1] = test_data
    
    
    def split_train_test_for_TUQ(self, pairs: list, labels: list) -> list:
        test_pairs, train_pairs = [pair[i] for i, pair in enumerate(pairs)], [pair[:i]+pair[i+1:] for i, pair in enumerate(pairs)]
        test_labels, train_labels = [label[i] for i, label in enumerate(labels)], [label[:i]+label[i+1:] for i, label in enumerate(labels)]
        
        np_test_pairs = np.array(test_pairs, dtype = str)
        np_test_labels = np.array(test_labels, dtype = str)
        
        train_pairs = np.array(train_pairs, dtype = str)
        train_labels = np.array(train_labels, dtype = str)
        np_train_pairs = train_pairs.reshape(train_pairs.shape[0], train_pairs.shape[1]*train_pairs.shape[2])
        np_train_labels = train_labels.reshape(train_labels.shape[0], train_labels.shape[1]*train_labels.shape[2])
                
        return np_train_pairs, np_train_labels, np_test_pairs, np_test_labels
            
            
            
        return TUQset
    
    
    def encording(self, pairs: np.ndarray, labels: np.ndarray, way_type: str) -> list:
        #np.zeros((num_cross_val, 1), dtype = str)の分を削除
        pairs = np.delete(pairs, 0, 1).tolist()
        labels = np.delete(labels, 0, 1).tolist()
        
        pairs = list(map(self.bert_tokenizer.encode, pairs))
        labels = list(map(self.replace_way_type, labels, [way_type]*len(labels)))
        
        dataset = list(map(self.create_dataset, pairs, labels))
        return dataset
    
    def replace_way_type(self, labels: list, way_type: str) -> list:
        if way_type == "2-way":
            labels = list(map(self.two_way.get, labels))
        elif way_type == "4-way":
            labels = list(map(self.four_way.get, labels))
        else:
            labels = list(map(self.seven_way.get, labels))
            
        return labels
    
    def create_dataset(self, pairs: list, labels: list) -> TensorDataset:
        
        labels = np.array(labels, dtype = int)
        labels = labels - 1
        dataset = TensorDataset(pairs["input_ids"], pairs["attention_mask"], 
                            torch.from_numpy(labels))
        return dataset


# In[85]:


def build_hatsumon_data(zyouhou1_path: str, tokenizer) -> HatsumonDataGenerator:
    if not(os.path.exists(zyouhou1_path)):
        raise ValueError("Cannot find json file. Please run create_dataset_from_csv.py")
    generator = HatsumonDataGenerator(zyouhou1_path, tokenizer)
    return generator


# # In[86]:


# path = "./zyouhou1_hatsumon.json"
# zd = build_hatsumon_data(path, None)

