#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2019/4/23 10:19
@Author     : Li Shanlu
@File       : main.py
@Software   : PyCharm
@Description: Knowledge distillation.
You can use inception_resnet_v2 as bigModel and inception_resnet_v1 as smallModel,
or inception_resnet_v1 as bigModel and mobile_v2 as smallModel.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import sys
import argparse
import model
from utils import data_loader,utils
import numpy as np
import random
import ipdb

seed = int(os.getenv("SEED", 12))
tf.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)


def check_and_makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def convert_str_to_bool(text):
    if text.lower() in ["true", "yes", "y", "1"]:
        return True
    else:
        return False


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--display_step', type=int, default=500)
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoint")
    parser.add_argument('--log_dir', type=str, default="logs")
    parser.add_argument('--gpu', type=int, default=None, choices=[None, 0, 1, 2, 3])

    # Training Parameters
    parser.add_argument('--train_data_dir', type=str, default='/data/lishanlu/dataset/train/train_data_182')
    parser.add_argument('--load_teacher_from_checkpoint', type=str, default="false")
    parser.add_argument('--load_teacher_checkpoint_dir', type=str, default=None)
    parser.add_argument('--model_type', type=str, default="teacher", choices=["teacher", "student"])
    parser.add_argument('--num_steps', type=int, default=500000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--distill_type', type=str, default='label', choices=['label', 'embedding'])

    # validation
    parser.add_argument('--unit_test_dir', type=str, default='')
    parser.add_argument('--test_batch_size', type=int, default=100)

    # Model Parameters
    parser.add_argument('--bigModel_name', type=str, default="net.inception_resnet_v1",
                        choices=["net.inception_resnet_v2", "net.inception_resnet_v1"])
    parser.add_argument('--smallModel_name', type=str, default="net.mobilenet_v2",
                        choices=["net.inception_resnet_v1", "net.mobilenet_v2"])
    parser.add_argument('--initial_learning_rate', type=float, default=0.01)
    parser.add_argument('--learning_rate_decay_steps', type=int, default=20000)
    parser.add_argument('--learning_rate_decay_factor', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--dropout_prob', type=float, default=0.8)
    parser.add_argument('--center_loss_factor', type=float, default=1e-2)
    parser.add_argument('--center_loss_alfa', type=float, default=0.9)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=2e-4)

    return parser


def setup(args):
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % (args.gpu)

    args.load_teacher_from_checkpoint = convert_str_to_bool(args.load_teacher_from_checkpoint)
    check_and_makedir(args.log_dir)
    check_and_makedir(args.checkpoint_dir)


def main():
    parser = get_parser()
    args = parser.parse_args()
    setup(args)
    # read train data
    train_set = utils.get_dataset(args.train_data_dir)
    nrof_classes = len(train_set)

    # read validation data
    print('unit test directory: %s' % args.unit_test_dir)
    unit_test_paths, unit_actual_issame = utils.get_val_paths(os.path.expanduser(args.unit_test_dir))
    nrof_test_img = len(unit_test_paths)
    unit_issame_label = np.zeros(nrof_test_img)
    for i in range(len(unit_actual_issame)):
        unit_issame_label[2*i] = unit_actual_issame[i]
        unit_issame_label[2*i+1] = unit_actual_issame[i]
    unit_issame_label = np.asarray(unit_issame_label, dtype=np.int32)
    # Get a list of image paths and their labels
    image_list, label_list = utils.get_image_paths_and_labels(train_set)
    assert len(image_list) > 0, 'The dataset should not be empty'

    print('Total number of train classes: %d' % nrof_classes)
    print('Total number of train examples: %d' % len(image_list))
    print("number of validation examples: %d" % nrof_test_img)
    #ipdb.set_trace()
    train_dataset = data_loader.DataLoader(image_list, label_list, [160, 160], nrof_classes)
    validation_dataset = data_loader.DataLoader(unit_test_paths, unit_issame_label, [160, 160])
    tf.reset_default_graph()
    if args.model_type == "student":
        teacher_model = None
        if args.load_teacher_from_checkpoint:
            teacher_model = model.BigModel(args, "teacher", nrof_classes, nrof_test_img)
            teacher_model.start_session()
            teacher_model.load_model_from_file(args.load_teacher_checkpoint_dir)
            print("Verify Teacher State before Training Student")
            teacher_model.run_inference(validation_dataset, unit_actual_issame)
        student_model = model.SmallModel(args, "student", nrof_classes, nrof_test_img)
        student_model.start_session()
        student_model.train(train_dataset, validation_dataset, unit_actual_issame, teacher_model)

        # Testing student model on the best model based on validation set
        student_model.load_model_from_file(args.checkpoint_dir)
        student_model.run_inference(validation_dataset, unit_actual_issame)

        if args.load_teacher_from_checkpoint:
            print("Verify Teacher State After Training student Model")
            teacher_model.run_inference(validation_dataset, unit_actual_issame)
            teacher_model.close_session()
        student_model.close_session()
    else:
        teacher_model = model.BigModel(args, "teacher", nrof_classes, nrof_test_img)
        teacher_model.start_session()
        teacher_model.train(train_dataset, validation_dataset, unit_actual_issame)

        # Testing teacher model on the best model based on validation set
        teacher_model.load_model_from_file(args.checkpoint_dir)
        teacher_model.run_inference(validation_dataset, unit_actual_issame)
        teacher_model.close_session()


if __name__ == '__main__':
    main()
