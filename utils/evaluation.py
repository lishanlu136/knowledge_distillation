#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2019/4/23 14:10
@Author     : Li Shanlu
@File       : evaluation.py
@Software   : PyCharm
@Description: 用于计算两个特征的相似度，向量点积
"""
import numpy as np


def calculate_acc_dot_product(thresholds, embeddings1, embeddings2, actual_issame):
    accs, recalls, fprs, precisions = np.zeros(len(thresholds)), np.zeros(len(thresholds)),\
                                      np.zeros(len(thresholds)), np.zeros(len(thresholds))
    dot_res_same = np.sum(embeddings1[actual_issame] * embeddings2[actual_issame], 1)
    dot_res_nsame = np.sum(embeddings1[~actual_issame] * embeddings2[~actual_issame], 1)

    for idx, threshold in enumerate(thresholds):
        tp = np.sum(dot_res_same >= threshold)
        fn = np.sum(actual_issame) - tp  # 事实同一人 预测为不同人

        tn = np.sum(dot_res_nsame < threshold)
        fp = np.sum(~actual_issame) - tn  # 事实不同人 预测为同一人

        accs[idx] = acc = (tp + tn) * 1.0 / (tp + fp + tn + fn)
        recalls[idx] = recall = tp * 1.0 / (tp + fn)  # 同一个人的人脸被判定正确的比率
        fprs[idx] = fpr = fp * 1.0 / (fp + tn)  # 不是同一个人被判定为同一人的比率
        precisions[idx] = precision = tp * 1.0 / (tp + fp + 0.000001)

        print("[calculate_acc_dot_product-%d]:threshold:%1.3f,recall:%1.3f,precision:%1.3f,fpr:%1.3f,acc:%1.3f"
              % (idx, threshold, recall, precision, fpr, acc))
    max_acc_idx = np.argmax(accs)
    print("[best_threshold_index:%d],threshold:%1.3f,acc:%1.3f,recall:%1.3f,precision:%1.3f,fpr:%1.3f" %
          (max_acc_idx, thresholds[max_acc_idx], accs[max_acc_idx], recalls[max_acc_idx], precisions[max_acc_idx],
           fprs[max_acc_idx]))

    # 返回准确率最大时预测错样本的idx
    best_threshold = thresholds[max_acc_idx]
    dot_product_all = np.sum(embeddings1 * embeddings2, 1)
    same_idxs = np.where(actual_issame)
    n_same_idxs = np.where(~actual_issame)
    bigger_idxs = np.where(dot_product_all >= best_threshold)
    less_idxs = np.where(dot_product_all < best_threshold)
    fp_idxs = np.intersect1d(bigger_idxs[0], n_same_idxs[0])  # fp 不同人 大于阈值的索引
    fn_idxs = np.intersect1d(less_idxs[0], same_idxs[0])  # fn 同一个人 小于阈值的索引
    return thresholds[max_acc_idx], accs[max_acc_idx], recalls[max_acc_idx], fprs[max_acc_idx],\
           precisions[max_acc_idx], dot_product_all, fp_idxs, fn_idxs