#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2019/4/23 16:05
@Author     : Li Shanlu
@File       : utils.py
@Software   : PyCharm
@Description:
"""

import os
import numpy as np
import random


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = os.listdir(path_exp)
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat


def get_val_paths(unit_data_dir, batch_size=100, num=1):
    path_list = []
    issame_list = []
    dir_ary = unit_data_dir.split(",")
    imgs_all_correct = []
    imgs_all_error = []
    for unit_dir in dir_ary:
        print("got dir :"+unit_dir)
        for dir in os.listdir(unit_dir):
            if dir == "correct":
                img_pairs = get_sub_img(os.path.join(unit_dir, dir))
                print("imgs in correct dir:" + os.path.join(unit_dir, dir)+" len:"+ str(len(img_pairs)))
                imgs_all_correct += img_pairs
            elif dir == "error":
                img_pairs = get_sub_img(os.path.join(unit_dir, dir))
                print("imgs in error dir:" + os.path.join(unit_dir, dir)+" len:" + str(len(img_pairs)))
                imgs_all_error += img_pairs
            else:
                # os.rmdir(os.path.join(unit_dir, dir))
                print("remove unit image path:"+os.path.join(unit_dir, dir))

    # 取相同样本数
    min_list_len = min(len(imgs_all_correct), len(imgs_all_error))
    print("imgs_all_correct:"+str(len(imgs_all_correct))+" imgs_all_error:"+
          str(len(imgs_all_error))+" min_list_len:"+str(min_list_len))
    # 在取样本之前打乱correct和error的pair顺序
    # correct
    correct_idx = range(int(len(imgs_all_correct)/2))
    random.shuffle(correct_idx, random.seed(20))
    shuffled_correct = []
    for x in correct_idx:
        shuffled_correct += [imgs_all_correct[x * 2], imgs_all_correct[x * 2 + 1]]
    # error
    error_idx = range(int(len(imgs_all_error)/2))
    random.shuffle(error_idx, random.seed(30))
    shuffled_error = []
    for x in error_idx:
        shuffled_error += [imgs_all_error[x * 2], imgs_all_error[x * 2 + 1]]
    actual_len = int(min_list_len/(batch_size*num)) * (batch_size*num)  # batch 的倍数
    print("actual batch len:"+str(actual_len))
    path_list += shuffled_correct[:actual_len]
    issame_list += [True]*(int(actual_len/2))
    path_list += shuffled_error[:actual_len]
    issame_list += [False]*(int(actual_len/2))
    shuffled_idx = range(len(issame_list))
    random.shuffle(shuffled_idx, random.seed(10))
    path_list_final = []
    for x in shuffled_idx:
        path_list_final += [path_list[x*2], path_list[x*2+1]]
    print("path list:", len(path_list_final), "issame_list:", len(issame_list))
    return path_list_final, np.array(issame_list)[shuffled_idx]


def get_sub_img(cur_dir):
    imgs_all = []
    for img_dir in os.listdir(cur_dir):
        imgs = []
        if os.path.isfile(os.path.join(cur_dir, img_dir)):
            print("skip file:"+os.path.join(cur_dir, img_dir))
            continue
        for img in os.listdir(os.path.join(cur_dir, img_dir)):
            imgs.append(os.path.join(cur_dir, img_dir, img))
        if len(imgs) == 2:
            # print("got pair:"+str(imgs))
            imgs_all += imgs
        else:
            print("skip path:"+os.path.join(cur_dir, img_dir, img))
    return imgs_all