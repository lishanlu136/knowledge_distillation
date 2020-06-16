#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2019/3/14 11:00
@Author     : Li Shanlu
@File       : freeze_student_model.py
@Software   : PyCharm
@Description:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import graph_util
import tensorflow as tf
import argparse
import os
import sys
import importlib
from tensorflow.python import pywrap_tensorflow
from six.moves import xrange
import ipdb


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
    network = importlib.import_module(args.model_def)
    g = tf.Graph()
    with g.as_default():
        image_input = tf.placeholder(dtype=tf.float32, shape=[None, 160, 160, 3], name="image_input")
        image_input = tf.identity(image_input, 'input')    #加上这句可以减小显存
        keep_probability = 1.0
        #phase_train = False
        phase_train = tf.placeholder(tf.bool, name='phase_train')
        embedding_size = 128
        weight_decay = 0.0
        # mobilenet_v2_1
        prelogits, _ = network.inference(image_input, keep_probability,
                                         phase_training=phase_train,
                                         bottleneck_layer_size=embedding_size,
                                         weight_decay=weight_decay)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        #all_vars = tf.trainable_variables()
        all_vars = tf.global_variables()
        print("+++++++++++++++++++++++++++++++")
        print("All eval trainable var:")
        for v in all_vars:
            print(v.name)
        print("+++++++++++++++++++++++++++++++")
        saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False), graph=g) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            load_model_from_file(sess, all_vars, args.dir_to_ckpt)
            #ckpt_name = os.path.join(args.output_file, 'distillation_mobilenet_v2.ckpt')
            #saver.save(sess, ckpt_name)

            # Retrieve the protobuf graph definition and fix the batch norm nodes
            input_graph_def = sess.graph.as_graph_def()
            # print(input_graph_def.node)

            # Freeze the graph def
            output_graph_def = freeze_graph_def(sess, input_graph_def, 'embeddings')
            # output_graph_def = freeze_graph_def(sess, input_graph_def, 'Bottleneck/BatchNorm/Reshape_1')

        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(args.output_file, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph: %s" % (len(output_graph_def.node), args.output_file))


def load_model_from_file(sess, vars, load_path):
    ckpt = tf.train.get_checkpoint_state(load_path)
    # ipdb.set_trace()
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt.model_checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            if key.startswith('student/MobilenetV2') or key.startswith('student/Bottleneck'):
                #print("tensor_name: ", key)
                weight = reader.get_tensor(key)
                #print("weight:", weight)
                for var in vars:
                    if var.name[:-2] in key:
                        sess.run(var.assign(weight))
                        print("tensor_name: ", key)
                        print("%s assign success." % var.name[:-2])
            else:
                continue
    else:
        print("Created model with fresh parameters.")
        sess.run(tf.global_variables_initializer())


def freeze_graph_def(sess, input_graph_def, output_node_names):
    for node in input_graph_def.node:
        #print(node)
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
    # Get the list of important nodes
    whitelist_names = []
    for node in input_graph_def.node:
        if (node.name.startswith('MobilenetV2') or node.name.startswith('embeddings') or
                node.name.startswith('phase_train') or node.name.startswith('Bottleneck')):
            whitelist_names.append(node.name)
    """
    print(">>>>>>> whitelist_names:")
    for i in whitelist_names:
        print(i)
    """

    # Replace all the variables in the graph with constants of the same values
    output_graph_def = graph_util.convert_variables_to_constants(
    sess, input_graph_def, output_node_names.split(","),
    variable_names_whitelist=whitelist_names)
    return output_graph_def


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_idx', type=str, help='gpu indexs', default='0')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='net.mobilenet_v2')
    parser.add_argument('--dir_to_ckpt', type=str,
                    help='Path to the directory include checkpoint (ckpt) file containing model parameters')
    parser.add_argument('--output_file', type=str,
                    help='Filename for the exported graphdef protobuf (.pb)')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))