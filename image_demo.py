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
import tensorflow as tf
from PIL import Image

from absl import app, flags

DEFAULT_PB = './models/yolov3_800_66_chn_test_loss_2.4417.ckpt-259.pb'

# 定义输入参数
FLAGS = flags.FLAGS
flags.DEFINE_string('image_path', './docs/images/road.jpeg', 'the image to predicate')
flags.DEFINE_string('pb_name', DEFAULT_PB, 'the model(pb) to predicate image')

def predicate_from_file(image_path, pb_file):
    original_image = cv2.imread(image_path)
    return predicate(original_image, pb_file)

def predicate(original_image, pb_file):
    if pb_file == None:
        pb_file = DEFAULT_PB
    
    return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    num_classes     = 66
    input_size      = 608
    graph           = tf.Graph()

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...]

    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)


    with tf.compat.v1.Session(graph=graph) as sess:
        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})

    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
    bboxes = utils.nms(bboxes, 0.45, method='nms')
    return bboxes, original_image

def main(argv):
    del argv
    bboxes, original_image = predicate_from_file(FLAGS.image_path, FLAGS.pb_name)
    image = utils.draw_bbox(original_image, bboxes)
    image = Image.fromarray(image)
    image.show()

if __name__ == '__main__':
    app.run(main)
