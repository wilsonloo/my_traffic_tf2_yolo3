#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-01-20 16:06:06
#   Description :
#
#================================================================

import cv2
import numpy as np
import core.utils as utils
from absl import app, flags

# lws
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from PIL import Image

# 定义输入参数
FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt_name', None, 'the checkpoint to be exported as (pb)model')
flags.DEFINE_string('pb_name', None, 'the freezon pb file')
flags.DEFINE_string('model_name', None, 'the (pb)model name')

def export(ckpt_name, pb_name, model_name):
    # 快照文件
    ckpt_path = "./checkpoint/" + ckpt_name
    # 导入模型pb，构建graph
    model_path = "./models/" + pb_name
    # 导出的savedModel 路径
    saved_model_dir = "./models/" + model_name
    
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.Graph()
    
    with graph.as_default():
        sess = tf.compat.v1.Session()
        
        # 恢复指定的tensor
        tf.import_graph_def(graph_def, name="import")
        input_image = graph.get_tensor_by_name('import/input/input_data:0')
        pred_sbbox = graph.get_tensor_by_name('import/pred_sbbox/concat_2:0')
        pred_mbbox = graph.get_tensor_by_name('import/pred_mbbox/concat_2:0')
        pred_lbbox = graph.get_tensor_by_name('import/pred_lbbox/concat_2:0')

        print("Restoring checkpoint from:", ckpt_path)
        new_saver = tf.compat.v1.train.import_meta_graph(ckpt_path+".meta")
        new_saver.restore(sess, ckpt_path)
        
        print("Exporting trained model to", saved_model_dir)
        # 构造定义一个builder， 并指定模型的输出路径
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(saved_model_dir)
        
        # 声明模型的input/output
        tensor_info_input      = tf.compat.v1.saved_model.utils.build_tensor_info(input_image)
        tensor_info_pred_sbbox = tf.compat.v1.saved_model.utils.build_tensor_info(pred_sbbox)
        tensor_info_pred_mbbox = tf.compat.v1.saved_model.utils.build_tensor_info(pred_mbbox)
        tensor_info_pred_lbbox = tf.compat.v1.saved_model.utils.build_tensor_info(pred_lbbox)
        prediction_signature = (
                tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                    inputs={"image": tensor_info_input}, 
                    outputs={"pred_sbbox": tensor_info_pred_sbbox, 
                        "pred_mbbox": tensor_info_pred_mbbox, 
                        "pred_lbbox": tensor_info_pred_lbbox}, 
                    method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME)) 
        builder.add_meta_graph_and_variables(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], 
                signature_def_map={'predict_image': prediction_signature})
        builder.save()
        print("Export Done.")


def main(argv):
    del argv
    if not FLAGS.ckpt_name:
        print('[ERROR] please set ckpt_name')
        raise()
    
    if not FLAGS.pb_name:
        print('[ERROR] please set pb_name')
        raise()
         
    if not FLAGS.model_name:
        print('[ERROR] please set model_name')
        raise()
    
    export(FLAGS.ckpt_name, FLAGS.pb_name, FLAGS.model_name)

if __name__ == '__main__':
    app.run(main)
