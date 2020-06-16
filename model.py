#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2019/4/23 10:19
@Author     : Li Shanlu
@File       : model.py
@Software   : PyCharm
@Description:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import sys
f_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f_path+"/..")
import importlib
from utils import losses,evaluation
import numpy as np
from tensorflow.python import pywrap_tensorflow
import ipdb


class BigModel:
    def __init__(self, args, model_type, num_train_classes, nrof_test_images):
        self.initial_learning_rate = args.initial_learning_rate
        self.learning_rate_decay_steps = args.learning_rate_decay_steps
        self.learning_rate_decay_factor = args.learning_rate_decay_factor
        self.num_steps = args.num_steps
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.display_step = args.display_step
        self.center_loss_factor = args.center_loss_factor
        self.center_loss_alfa = args.center_loss_alfa
        self.num_classes = num_train_classes
        self.nrof_test_images = nrof_test_images
        self.dropout_prob = args.dropout_prob
        self.embedding_size = args.embedding_size
        self.weight_decay = args.weight_decay
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_file = "bigmodel"
        self.temperature = args.temperature
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file + ".ckpt")
        self.log_dir = os.path.join(args.log_dir, self.checkpoint_file)
        self.model_type = model_type
        self.model_name = args.bigModel_name

        self.build_model()
        self.saver = tf.train.Saver()

    def build_model(self):
        network = importlib.import_module(self.model_name)
        with tf.name_scope("%s" % self.model_type), tf.variable_scope("%s" % self.model_type):
            self.global_step = tf.Variable(0, trainable=False)
            self.x_placeholder = tf.placeholder(tf.float32, [None, 160, 160, 3],
                                                name="%s_%s" % (self.model_type, "x_input"))
            self.y_placeholder = tf.placeholder(tf.int64, [None,], name="%s_%s" % (self.model_type, "y_input"))
            self.dropout_prob_placeholder = tf.placeholder(tf.float32, name="%s_%s" % (self.model_type, "dropout_prob"))
            self.phase_train_placeholder = tf.placeholder(tf.bool, name="%s_%s" % (self.model_type, "phase_train"))
            self.softmax_temperature_placeholder = tf.placeholder(tf.float32,
                                                                  name="%s_%s" % (self.model_type, "softmax_temp"))
            self.lr_placeholder = tf.placeholder(tf.float32, name="%s_%s" % (self.model_type, "learning_rate"))
            
            # build graph
            prelogits, _ = network.inference(self.x_placeholder, self.dropout_prob_placeholder,
                                             phase_train=self.phase_train_placeholder,
                                             bottleneck_layer_size=self.embedding_size,
                                             weight_decay=self.weight_decay)
            logits = slim.fully_connected(prelogits, self.num_classes, activation_fn=None,
                                          weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                          weights_regularizer=slim.l2_regularizer(self.weight_decay),
                                          scope='Logits', reuse=False)/self.softmax_temperature_placeholder
            self.embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name="%s_%s" % (self.model_type, 'embeddings'))

            with tf.name_scope("%s_prediction" % (self.model_type)),\
                                tf.variable_scope("%s_prediction" % (self.model_type)):
                self.prediction = tf.nn.softmax(logits)
                # Evaluate model
                correct_pred = tf.equal(tf.argmax(self.prediction, 1), self.y_placeholder)
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            with tf.name_scope("%s_optimization" % (self.model_type)),\
                                tf.variable_scope("%s_optimization" % (self.model_type)):
                # Define loss and optimizer
                # center loss
                if self.center_loss_factor > 0.0:
                    prelogits_center_loss, _ = losses.center_loss(prelogits, self.y_placeholder,
                                                                  self.center_loss_alfa, self.num_classes)
                    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                         prelogits_center_loss * self.center_loss_factor)
                self.cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                         logits=logits, labels=self.y_placeholder))
                tf.add_to_collection('losses', self.cross_entropy_mean)

                # Calculate the total losses
                self.regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                self.loss_op = tf.add_n([self.cross_entropy_mean] + self.regularization_losses, name='total_loss')
                self.learning_rate = tf.train.exponential_decay(self.lr_placeholder, self.global_step,
                                                                self.learning_rate_decay_steps,
                                                                self.learning_rate_decay_factor, staircase=True)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                                          decay=0.9, momentum=0.9, epsilon=1.0)
                    self.train_op = optimizer.minimize(self.loss_op, global_step=self.global_step)

            with tf.name_scope("%s_summarization" % (self.model_type)),\
                               tf.variable_scope("%s_summarization" % (self.model_type)):
                tf.summary.scalar("total_loss", self.loss_op)
                tf.summary.scalar("learning_rate", self.learning_rate)

                for var in tf.trainable_variables():
                    tf.summary.histogram(var.name, var)

                def my_merging_function(scope_str):
                    with tf.name_scope("%s_%s" % (self.model_type, "summary_merger")), tf.variable_scope(
                            "%s_%s" % (self.model_type, "summary_merger")):
                        from tensorflow.python.framework import ops as _ops
                        key = _ops.GraphKeys.SUMMARIES
                        summary_ops = _ops.get_collection(key, scope=scope_str)
                        if not summary_ops:
                            return None
                        else:
                            return tf.summary.merge(summary_ops)

                self.merged_summary_op = my_merging_function(self.model_type)

        # print trainable vars
        self.all_vars = tf.trainable_variables()
        print("bigmodel trainable vars:\n+++++++++++++++++++++++++++++")
        for var in self.all_vars:
            print(var.name)
        print("+++++++++++++++++++++++++++++")

    def start_session(self):
        self.sess = tf.Session()

    def close_session(self):
        self.sess.close()

    def train(self, train_dataset, val_dataset, actual_issame):
        # ipdb.set_trace()
        element, init_op = train_dataset.data_batch(augment=True,
                                                    shuffle=True,
                                                    batch_size=self.batch_size,
                                                    repeat_times=100,
                                                    num_threads=4,
                                                    buffer=120000)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(init_op)
        print("Starting Training...\n")
        train_summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)
        max_accuracy = 0
        for step in range(self.num_steps + 1):
            element_data = self.sess.run(element)
            # ipdb.set_trace()
            _, summary = self.sess.run([self.train_op, self.merged_summary_op],
                                       feed_dict={self.x_placeholder: element_data[0],
                                                  self.y_placeholder: element_data[1],
                                                  self.dropout_prob_placeholder: self.dropout_prob,
                                                  self.phase_train_placeholder: True,
                                                  self.softmax_temperature_placeholder: self.temperature,
                                                  self.lr_placeholder: self.initial_learning_rate})

            if (step % self.display_step) == 0 or step == 1:
                # ipdb.set_trace()
                # Calculate Validation accuracy
                acc = self.run_inference(val_dataset, actual_issame)
                if acc > max_accuracy:
                    save_path = self.saver.save(self.sess, self.checkpoint_path)
                    print("Model checkpointed to %s " % save_path)
                    max_accuracy = acc
                print("step:" + str(step) + ", Validation Accuracy= " + "{:.3f}".format(acc))
                val_summary = tf.Summary()
                val_summary.value.add(tag='val/acc_dot', simple_value=acc)
                train_summary_writer.add_summary(val_summary, step)
                train_summary_writer.add_summary(summary, step)
        else:
            # Final Evaluation and checkpoint before training ends
            acc = self.run_inference(val_dataset, actual_issame)
            if acc > max_accuracy:
                save_path = self.saver.save(self.sess, self.checkpoint_path)
                print("Model checkpointed to %s " % save_path)

        train_summary_writer.close()

        print("Optimization Finished!")

    def predict(self, data_x, temperature=1.0):
        return self.sess.run(self.prediction,
                             feed_dict={self.x_placeholder: data_x,
                                        self.dropout_prob_placeholder: 1.0,
                                        self.phase_train_placeholder: False,
                                        self.softmax_temperature_placeholder: temperature})
    
    def get_embedding(self, data_x):
        return self.sess.run(self.embeddings, feed_dict={self.x_placeholder: data_x,
                                                         self.phase_train_placeholder: False,
                                                         self.dropout_prob_placeholder: 1.0})

    def run_inference(self, test_dataset, actual_issame):
        element, init_op = test_dataset.data_batch(augment=True,
                                                   shuffle=False,
                                                   batch_size=self.test_batch_size,
                                                   repeat_times=1,
                                                   num_threads=4,
                                                   buffer=1)
        self.sess.run(init_op)
        emb_array = np.zeros((self.nrof_test_images, self.embedding_size))
        i = 0
        try:
            while True:
                start_index = i * self.test_batch_size
                end_index = min((i + 1) * self.test_batch_size, self.nrof_test_images)
                element_data = self.sess.run(element)
                feed_dict = {self.x_placeholder: element_data[0],
                             self.phase_train_placeholder: False,
                             self.dropout_prob_placeholder: 1.0}
                emb = self.sess.run([self.embeddings], feed_dict=feed_dict)
                emb_array[start_index:end_index, :] = emb[0]
                i += 1
        except tf.errors.OutOfRangeError:
            print("end.")
        # evaluation
        embeddings1 = emb_array[0::2]
        embeddings2 = emb_array[1::2]
        dot_product_threshold = np.arange(0.0, 1.0, 0.001)
        best_threshold, acc_dot, recall, fpr_dot, precision_dot, dot_product_all, fp_idxs, fn_idxs = \
            evaluation.calculate_acc_dot_product(dot_product_threshold, embeddings1, embeddings2, np.asarray(actual_issame))
        print("acc dot:", acc_dot)
        return acc_dot

    def load_model_from_file(self, load_path):
        ckpt = tf.train.get_checkpoint_state(load_path)
        # ipdb.set_trace()
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            reader = pywrap_tensorflow.NewCheckpointReader(ckpt.model_checkpoint_path)
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in var_to_shape_map:
                print("tensor_name: ", key)
                weight = reader.get_tensor(key)
                for var in self.all_vars:
                    if key in var.name:
                        self.sess.run(var.assign(weight))
            #self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())


class SmallModel:
    def __init__(self, args, model_type, num_train_classes, nrof_test_images):
        self.initial_learning_rate = args.initial_learning_rate
        self.learning_rate_decay_steps = args.learning_rate_decay_steps
        self.learning_rate_decay_factor = args.learning_rate_decay_factor
        self.num_steps = args.num_steps
        self.dropout_prob = args.dropout_prob
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.display_step = args.display_step
        self.center_loss_factor = args.center_loss_factor
        self.center_loss_alfa = args.center_loss_alfa
        self.num_classes = num_train_classes
        self.nrof_test_images = nrof_test_images
        self.embedding_size = args.embedding_size
        self.weight_decay = args.weight_decay
        self.temperature = args.temperature
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_file = "smallmodel"
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file)
        self.max_checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file + "max")
        self.log_dir = os.path.join(args.log_dir, self.checkpoint_file)
        self.model_type = model_type
        self.model_name = args.smallModel_name
        self.distill_type = args.distill_type

        self.build_model()
        self.saver = tf.train.Saver()

    def build_model(self):
        network = importlib.import_module(self.model_name)
        with tf.name_scope("%s" % self.model_type), tf.variable_scope("%s" % self.model_type):
            self.global_step = tf.Variable(0, trainable=False)
            self.x_placeholder = tf.placeholder(tf.float32, [None, 160, 160, 3],
                                                name="%s_%s" % (self.model_type, "x_input"))
            self.y_placeholder = tf.placeholder(tf.int64, [None,], name="%s_%s" % (self.model_type, "y_input"))
            self.dropout_prob_placeholder = tf.placeholder(tf.float32, name="%s_%s" % (self.model_type, "dropout_prob"))
            self.phase_train_placeholder = tf.placeholder(tf.bool, name="%s_%s" % (self.model_type, "phase_train"))
            self.flag_placeholder = tf.placeholder(tf.bool, None, name="%s_%s" % (self.model_type, "flag"))
            self.soft_y_placeholder = tf.placeholder(tf.float32, [None, self.num_classes],
                                                     name="%s_%s" % (self.model_type, "soft_y"))
            self.soft_embedding_placeholder = tf.placeholder(tf.float32, [None, self.embedding_size],
                                                             name="%s_%s" % (self.model_type, "soft_embedding"))
            self.softmax_temperature_placeholder = tf.placeholder(tf.float32,
                                                                  name="%s_%s" % (self.model_type, "softmax_temp"))
            self.lr_placeholder = tf.placeholder(tf.float32, name="%s_%s" % (self.model_type, "learning_rate"))
            prelogits, _ = network.inference(self.x_placeholder,
                                             self.dropout_prob_placeholder,
                                             phase_train=self.phase_train_placeholder,
                                             bottleneck_layer_size=self.embedding_size,
                                             weight_decay=self.weight_decay)
            logits = slim.fully_connected(prelogits, self.num_classes, activation_fn=None,
                                          weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                          weights_regularizer=slim.l2_regularizer(self.weight_decay),
                                          scope='Logits', reuse=False)

            self.embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name="%s_%s" % (self.model_type, "embeddings"))
            # ipdb.set_trace()
            with tf.name_scope("%s_prediction" % (self.model_type)), tf.variable_scope("%s_prediction" % (
                    self.model_type)):
                self.prediction = tf.nn.softmax(logits)
                self.correct_pred = tf.equal(tf.argmax(logits, 1), self.y_placeholder)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

            with tf.name_scope("%s_optimization" % (self.model_type)), tf.variable_scope("%s_optimization" % (
                    self.model_type)):
                # Define loss and optimizer
                # center loss
                if self.center_loss_factor > 0.0:
                    prelogits_center_loss, _ = losses.center_loss(prelogits, self.y_placeholder,
                                                             self.center_loss_alfa, self.num_classes)
                    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                         prelogits_center_loss * self.center_loss_factor)
                self.cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                         logits=logits, labels=self.y_placeholder))
                tf.add_to_collection('losses', self.cross_entropy_mean)
                regular_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                self.regular_losses = [v for v in regular_loss if not (v.name.startswith('teacher/teacher/'))]
                self.total_loss = tf.add_n([self.cross_entropy_mean] + self.regular_losses, name='total_loss')
                if self.distill_type == 'label':
                    self.loss_op_soft = tf.cond(self.flag_placeholder,
                                                true_fn=lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                                logits=logits / self.softmax_temperature_placeholder,
                                                labels=self.soft_y_placeholder)),
                                                false_fn=lambda: 0.0)
                elif self.distill_type == 'embedding':
                    self.loss_op_soft = tf.cond(self.flag_placeholder,
                                                true_fn=lambda: tf.reduce_sum(tf.square(tf.subtract(
                                                self.soft_embedding_placeholder, self.embeddings))) / self.batch_size,
                                                false_fn=lambda: 0.0)
                else:
                    print('Distill type error, please use correct type: label or embedding.')
                    self.loss_op_soft = 0.0

                self.total_loss += tf.square(self.softmax_temperature_placeholder) * self.loss_op_soft

                self.learning_rate = tf.train.exponential_decay(self.lr_placeholder, self.global_step,
                                                                self.learning_rate_decay_steps,
                                                                self.learning_rate_decay_factor, staircase=True)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.9, momentum=0.9,epsilon=1.0)
                    self.train_op = optimizer.minimize(self.total_loss, global_step=self.global_step)

            with tf.name_scope("%s_summarization" % (self.model_type)), tf.variable_scope("%s_summarization" % (
                    self.model_type)):
                tf.summary.scalar("loss_op_standard", self.cross_entropy_mean)
                tf.summary.scalar("total_loss", self.total_loss)
                tf.summary.scalar("learning_rate", self.learning_rate)

                small_trainable_vars = tf.trainable_variables()
                vars = [v for v in small_trainable_vars if not (v.name.startswith('teacher/teacher/'))]
                for var in vars:
                    tf.summary.histogram(var.name, var)

                def my_merging_function(scope_str):
                    with tf.name_scope("%s_%s" % (self.model_type, "summary_merger")), tf.variable_scope(
                            "%s_%s" % (self.model_type, "summary_merger")):
                        from tensorflow.python.framework import ops as _ops
                        key = _ops.GraphKeys.SUMMARIES
                        summary_ops = _ops.get_collection(key, scope=scope_str)
                        if not summary_ops:
                            return None
                        else:
                            return tf.summary.merge(summary_ops)

                self.merged_summary_op = my_merging_function(self.model_type)

        # print trainable vars
        all_vars = tf.trainable_variables()
        print("+++++++++++++++++++++++++++++")
        for var in all_vars:
            print(var.name)
        print("+++++++++++++++++++++++++++++")

    def start_session(self):
        self.sess = tf.Session()

    def close_session(self):
        self.sess.close()

    def train(self, train_dataset, val_dataset, actual_issame, teacher_model=None):
        teacher_flag = False
        if teacher_model is not None:
            teacher_flag = True
        element, init_op = train_dataset.data_batch(augment=True,
                                                    shuffle=True,
                                                    batch_size=self.batch_size,
                                                    repeat_times=400,
                                                    num_threads=4,
                                                    buffer=120000)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(init_op)
        print("Starting Training ...\n")
        train_summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.small_sess.graph)
        max_accuracy = 0
        for step in range(self.num_steps+1):
            element_data = self.small_sess.run(element)
            # ipdb.set_trace()
            if teacher_flag:
                if self.distill_type == 'label':
                    soft_targets = teacher_model.predict(element_data[0], self.temperature)
                elif self.distill_type == 'embedding':
                    soft_targets = teacher_model.get_embedding(element_data[0])
            else:
                if self.distill_type == 'label':
                    soft_targets = self.sess.run(tf.one_hot(element_data[1], self.num_classes, 1.0, 0.0, dtype=tf.float32))
                elif self.distill_type == 'embedding':
                    soft_targets = np.zeros((self.batch_size, self.embedding_size))

            _, summary = self.sess.run([self.train_op, self.merged_summary_op],
                                        feed_dict={self.x_placeholder: element_data[0],
                                                   self.y_placeholder: element_data[1],
                                                   self.dropout_prob_placeholder: self.dropout_prob,
                                                   self.phase_train_placeholder: True,
                                                   self.soft_y_placeholder: soft_targets,
                                                   self.flag_placeholder: teacher_flag,
                                                   self.softmax_temperature_placeholder: self.temperature,
                                                   self.lr_placeholder: self.initial_learning_rate})

            if (step % self.display_step) == 0 or step == 1:
                acc = self.run_inference(val_dataset, actual_issame)
                if acc > max_accuracy:
                    save_path = self.saver.save(self.sess, self.checkpoint_path)
                    print("Model checkpointed to %s " % (save_path))
                    max_accuracy = acc
                print("step " + str(step) + ", Validation Accuracy= " + "{:.3f}".format(acc))
                val_summary = tf.Summary()
                val_summary.value.add(tag='val/acc_dot', simple_value=acc)
                train_summary_writer.add_summary(summary, step)
                train_summary_writer.add_summary(val_summary, step)
        else:
            # Final Evaluation and checkpoint before training ends
            acc = self.run_inference(val_dataset, actual_issame)
            if acc > max_accuracy:
                save_path = self.saver.save(self.sess, self.checkpoint_path)
                print("Model checkpointed to %s " % (save_path))
                max_accuracy = acc
            print("step " + str(step) + ", Validation Accuracy= " + "{:.3f}".format(acc))

        train_summary_writer.close()

        print("Optimization Finished!")

    def predict(self, data_x, temperature=1.0):
        return self.sess.run(self.prediction,
                             feed_dict={self.x_placeholder: data_x,
                                        self.flag_placeholder: False,
                                        self.dropout_prob_placeholder: 1.0,
                                        self.phase_train_placeholder: False,
                                        self.softmax_temperature_placeholder: temperature})

    def run_inference(self, test_dataset, actual_issame):
        element, init_op = test_dataset.data_batch(augment=True,
                                                   shuffle=False,
                                                   batch_size=self.test_batch_size,
                                                   repeat_times=1,
                                                   num_threads=4,
                                                   buffer=1)
        self.sess.run(init_op)
        emb_array = np.zeros((self.nrof_test_images, self.embedding_size))
        i = 0
        try:
            while True:
                start_index = i * self.test_batch_size
                end_index = min((i + 1) * self.test_batch_size, self.nrof_test_images)
                element_data = self.sess.run(element)
                feed_dict = {self.x_placeholder: element_data[0],
                             self.phase_train_placeholder: False,
                             self.dropout_prob_placeholder: 1.0}
                emb = self.sess.run([self.embeddings], feed_dict=feed_dict)
                emb_array[start_index:end_index, :] = emb[0]
                i += 1
        except tf.errors.OutOfRangeError:
            print("end.")
        # evaluation
        embeddings1 = emb_array[0::2]
        embeddings2 = emb_array[1::2]
        dot_product_threshold = np.arange(0.0, 1.0, 0.001)
        best_threshold, acc_dot, recall, fpr_dot, percision_dot, dot_product_all, fp_idxs, fn_idxs = \
            evaluation.calculate_acc_dot_product(dot_product_threshold, embeddings1, embeddings2, np.asarray(actual_issame))
        print("acc dot:", acc_dot)
        # save ckpt mode
        save_path = self.saver.save(self.sess, "train_results/distillation/studentckpt/smallmodel")
        print("save success, %s" % save_path)
        return acc_dot

    def load_model_from_file(self, load_path):
        ckpt = tf.train.get_checkpoint_state(load_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())
