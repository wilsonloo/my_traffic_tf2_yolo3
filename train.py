#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 17:50:26
#   Description :
#
#================================================================

import os
import re
import time
import math
import shutil
import numpy as np

#lws
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3 import YOLOV3
from core.config import cfg


class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale    = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes             = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes         = len(self.classes)
        self.learn_rate_init     = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end      = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs  = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods      = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight      = cfg.TRAIN.INITIAL_WEIGHT
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale  = 150
        self.train_logdir        = "./data/log/train"
        self.trainset            = Dataset('train')
        self.testset             = Dataset('test')
        self.steps_per_period    = len(self.trainset)

        tf_config = tf.ConfigProto(allow_soft_placement=True)
        
        #lws
        tf_config.gpu_options.allow_growth = True  

        self.sess                = tf.Session(config=tf_config)

        with tf.name_scope('define_input'):
            self.input_data   = tf.placeholder(dtype=tf.float32, name='input_data')
            self.label_sbbox  = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox  = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox  = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.trainable     = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope("define_loss"):
            self.model = YOLOV3(self.input_data, self.trainable)
            self.net_var = tf.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
                                                    self.label_sbbox,  self.label_mbbox,  self.label_lbbox,
                                                    self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                        dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant( (self.first_stage_epochs + self.second_stage_epochs)* self.steps_per_period,
                                        dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                    (1 + tf.cos(
                                        (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver  = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate",      self.learn_rate)
            tf.summary.scalar("giou_loss",  self.giou_loss)
            tf.summary.scalar("conf_loss",  self.conf_loss)
            tf.summary.scalar("prob_loss",  self.prob_loss)
            tf.summary.scalar("total_loss", self.loss)

            logdir = "./data/log/"
            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.summary.merge_all()
            self.summary_writer  = tf.summary.FileWriter(logdir, graph=self.sess.graph)


    def train(self):
        from_scratch = False
        self.sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
            from_scratch = False
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            from_scratch = True
            self.first_stage_epochs = 0
        
        # checkpont列表
        # 格式：checkpoint_path, loss, isLessLoss, epoch_to_bak]
        ckpt_list = []
        
        last_epoch = 0
        last_test_loss = None
        if not from_scratch:
            try:
                found = re.sub(r'.*yolov3_800_66_chn_test_loss_(.*)\.ckpt-(\d+)', r'\1:\2', self.initial_weight)
                foundList = found.split(':')
                last_test_loss, last_epoch = float(foundList[0]), int(foundList[1])
                ckpt_list.append([self.initial_weight+" @", last_test_loss])
            except:
                print("=> faield to capture the last epoch in %s" % self.initial_weight)
                last_epoch = 0

        # 最小的损失率
        min_loss_checkpoint = None
        min_loss_value = None
        min_loss_bak_epoch = None
        
        # 总的epoch
        total_epoch = self.first_stage_epochs+self.second_stage_epochs
        for loop_index in range(0, 6):
            last_epoch = last_epoch + loop_index * total_epoch
            for epoch in range(1, 1+total_epoch):
                global_epoch = last_epoch + epoch
                if epoch <= self.first_stage_epochs:
                    train_op = self.train_op_with_frozen_variables
                else:
                    train_op = self.train_op_with_all_variables

                pbar = tqdm(self.trainset)
                train_epoch_loss, test_epoch_loss = [], []

                for train_data in pbar:
                    _, summary, train_step_loss, global_step_val = self.sess.run(
                        [train_op, self.write_op, self.loss, self.global_step],feed_dict={
                                                    self.input_data:   train_data[0],
                                                    self.label_sbbox:  train_data[1],
                                                    self.label_mbbox:  train_data[2],
                                                    self.label_lbbox:  train_data[3],
                                                    self.true_sbboxes: train_data[4],
                                                    self.true_mbboxes: train_data[5],
                                                    self.true_lbboxes: train_data[6],
                                                    self.trainable:    True,
                    })

                    train_epoch_loss.append(train_step_loss)
                    self.summary_writer.add_summary(summary, global_step_val)
                    pbar.set_description("=> Epoch: %d(%d/%d) train loss: %.2f" % (global_epoch, epoch, total_epoch, train_step_loss))

                for test_data in self.testset:
                    test_step_loss = self.sess.run( self.loss, feed_dict={
                                                    self.input_data:   test_data[0],
                                                    self.label_sbbox:  test_data[1],
                                                    self.label_mbbox:  test_data[2],
                                                    self.label_lbbox:  test_data[3],
                                                    self.true_sbboxes: test_data[4],
                                                    self.true_mbboxes: test_data[5],
                                                    self.true_lbboxes: test_data[6],
                                                    self.trainable:    False,
                    })

                    test_epoch_loss.append(test_step_loss)

                # 记录该快照
                limit = 6
                
                # 如果列表未满，且小于最大的损失率才添加
                train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
                print("current-test-lost: %.4f" % test_epoch_loss)
                
                if test_epoch_loss != None and math.isnan(float(test_epoch_loss)) == False:
                    if last_test_loss == None or test_epoch_loss < (last_test_loss + 0.5): # 有容忍
                        # 如果ckpt_list未满，或者比最后一个还低才进行存档
                        if len(ckpt_list) < limit or test_epoch_loss < ckpt_list[limit-1][1]:
                            # 开始存档
                            ckpt_file = "./checkpoint/yolov3_800_66_chn_test_loss_%.4f.ckpt" % test_epoch_loss
                            log_start_time = time.localtime(time.time())
                            log_time = time.strftime('%Y-%m-%d %H:%M:%S', log_start_time)
                            print("=> Epoch: %d(%d/%d) Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                                            %(global_epoch, epoch, total_epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
                            self.saver.save(self.sess, ckpt_file, global_step=global_epoch)
                            
                            # 实际的快照名称
                            ckpt_file_path = ckpt_file+"-"+str(global_epoch)

                            # 记录最小损失的快照，为避免被框架定时删除，再次进行存档
                            if min_loss_value == None or test_epoch_loss < min_loss_value:
                                min_loss_value = test_epoch_loss
                                min_loss_checkpoint = ckpt_file_path
                                min_loss_bak_epoch = epoch + 5
                                
                             
                            # 记录该快照
                            ckpt_list.append([ckpt_file_path, test_epoch_loss])
                            ckpt_list.sort(key=lambda elem: elem[1])
                            if len(ckpt_list) > limit:
                                ckpt_list, remove_list = ckpt_list[:limit], ckpt_list[limit:]
                                for elem in remove_list:
                                    fname = elem[0]    
                                    loss = elem[1]
                                    if last_test_loss == None or loss > last_test_loss:
                                        cmd = "rm "+fname+"*"
                                        print("calling  --- ", cmd)
                                        os.system(cmd)
                        
                        if len(ckpt_list) > 0:
                            for k in range(len(ckpt_list)-1, -1, -1):
                                elem = ckpt_list[k]
                                print("current ranking:", elem[0])
                                
                        if min_loss_checkpoint != None and min_loss_value != None and min_loss_bak_epoch != None:
                            if epoch >= min_loss_bak_epoch:
                                os.system("cp "+min_loss_checkpoint+".* ./checkpoint/bak")
                                print(min_loss_checkpoint, " backuped")
                                min_loss_bak_epoch = None

if __name__ == '__main__': YoloTrain().train()




