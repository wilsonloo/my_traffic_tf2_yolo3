#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : freeze_graph.py
#   Author      : YunYang1994
#   Created date: 2019-03-20 15:57:33
#   Description :
#
#================================================================

#lws
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from absl import app, flags
from core.yolov3 import YOLOV3

# 定义输入参数
FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt_name', None, 'the checkpoint to be exported as (pb)model')
flags.DEFINE_string('model_name', None, 'the (pb)model name')

def export(ckpt_name, model_name):
    pb_file = "./models/"+model_name
    ckpt_file = "./checkpoint/"+ckpt_name

    output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]
    with tf.name_scope('input'):
        input_data = tf.placeholder(dtype=tf.float32, name='input_data')
    
    model = YOLOV3(input_data, trainable=False)
    print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)
    
    sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)
    
    converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                input_graph_def  = sess.graph.as_graph_def(),
                                output_node_names = output_node_names)
    
    with tf.gfile.GFile(pb_file, "wb") as f:
        f.write(converted_graph_def.SerializeToString())

def main(argv):
    del argv
    if not FLAGS.ckpt_name:
        print('[ERROR] please set ckpt_name')
        raise()
    
    if not FLAGS.model_name:
        print('[ERROR] please set model_name')
        raise()
    
    export(FLAGS.ckpt_name, FLAGS.model_name)

if __name__ == '__main__':
    app.run(main)
