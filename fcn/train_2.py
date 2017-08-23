#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   14.06.2017
#-------------------------------------------------------------------------------

import argparse
import math
import sys
import os

import tensorflow as tf
import numpy as np

from fcnvgg import FCNVGG
import utils
from tqdm import tqdm
from apmetrics import APMetrics
import shutil

#-------------------------------------------------------------------------------
# Parse the commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train the FCN')
parser.add_argument('--name', default='test',
                    help='project name')
parser.add_argument('--data-source', default='kitti',
                    help='data source')
parser.add_argument('--data-dir', default='/home/levin/workspace/carnd/semantic_segmentation/data',
                    help='data directory')
parser.add_argument('--vgg-dir', default='/home/levin/workspace/carnd/semantic_segmentation/data/vgg',
                    help='directory for the VGG-16 model')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=20,
                    help='batch size')
parser.add_argument('--tensorboard-dir', default="tb",
                    help='name of the tensorboard data directory')
parser.add_argument('--checkpoint-interval', type=int, default=50,
                    help='checkpoint interval')
args = parser.parse_args()



try:
    print('[i] Creating directory {}...'.format(args.name))
    if os.path.exists(args.name):
        shutil.rmtree(args.name) 
    os.makedirs(args.name)
    if os.path.exists(args.tensorboard_dir):
        shutil.rmtree(args.tensorboard_dir)
except (IOError) as e:
    print('[!]', str(e))
    sys.exit(1)




    
class TrainModel(object):
    def __init__(self):
        return
    def __disp_config(self):
        print('[i] Project name:         ', args.name)
        print('[i] Data source:          ', args.data_source)
        print('[i] Data directory:       ', args.data_dir)
        print('[i] VGG directory:        ', args.vgg_dir)
        print('[i] # epochs:             ', args.epochs)
        print('[i] Batch size:           ', args.batch_size)
        print('[i] Tensorboard directory:', args.tensorboard_dir)
        print('[i] Checkpoint interval:  ', args.checkpoint_interval)
        return
    def __load_data_source(self):
       
        print('load data source...')
        try:
            source_module = __import__('source_'+args.data_source)
            source    = getattr(source_module, 'get_source')()
            source.load_data(args.data_dir, 0.1)
            print('[i] # training samples:   ', source.num_training)
            print('[i] # validation samples: ', source.num_validation)
            print('[i] # classes:            ', source.num_classes)
            print('[i] Image size:           ', source.image_size)
            return source
        except (ImportError, AttributeError, RuntimeError) as e:
            print('[!] Unable to load data source:', str(e))
            sys.exit(1)
        return
   
        
    def run(self):
        self.__disp_config()
        source = self.__load_data_source()
        #-------------------------------------------------------------------------------
        # Create the network
        #-------------------------------------------------------------------------------
        with tf.Session() as sess:
            print('[i] Creating the model...')
            net = FCNVGG(sess, source.num_classes)
            net.build_from_vgg(args.vgg_dir, source.num_classes, progress_hook='tqdm')
        
            net.get_optimizer(net.labels)
            net.add_summary_nodes(args.tensorboard_dir)
            
            validation_imgs    = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            validation_img_summary_op = tf.summary.image('validation_img',validation_imgs)
            train_img_summary_op = tf.summary.image('train_img',validation_imgs)
            saver           = tf.train.Saver(max_to_keep=100)
           
        
            
            print('[i] Training...')
        
            
            utils.initialize_uninitialized_variables(sess)
            n_train_batches = int(math.ceil(source.num_training/args.batch_size))
            
            cur_step = -1
            step_num = n_train_batches * args.epochs
        
            for e in range(args.epochs):
              
                train_generator = source.train_generator(args.batch_size)
                 
             
                for x, y in train_generator:
                    cur_step += 1
                    feed = {net.image_input:  x,
                            net.labels:           y,
                            net.keep_prob:    0.5,
                            net.is_training: True}
                    
                    sess.run(net.reset_iou_op)
                    summary, _, loss_batch, _,label_mapper, img_classes = sess.run([net.merged, net.update_iou_op, 
                                                                                    net.loss, net.optimizer,net.label_mapper, net.classes], feed_dict=feed)
                    net.train_writer.add_summary(summary, cur_step)
                    iou, summary = sess.run([net.metric_iou__op, net.merged_update])
                    net.train_writer.add_summary(summary, cur_step)
                    print("step {}/{}: loss={}, iou={}".format(cur_step, step_num, loss_batch, iou))
                    #output trainig input image
                    if (cur_step+1) % 10 == 0:
                        val_imgs = x[:1,:,:,:]
                        val_img_labels = img_classes[:1, :, :]
                        val_img_labels_gt = label_mapper[:1, :, :]
                        imgs_inferred = utils.draw_labels_batch(val_imgs, val_img_labels, source.label_colors)
                        imgs_gt       = utils.draw_labels_batch(val_imgs, val_img_labels_gt, source.label_colors)
                        val_imgs = utils.convert_rgb_batch(val_imgs)
                        all_imgs = np.concatenate([val_imgs, imgs_gt, imgs_inferred], axis = 0)

                        summary = sess.run(train_img_summary_op,
                                                            feed_dict={validation_imgs: all_imgs})        
                        net.train_writer.add_summary(summary, cur_step)
                    #monitor inference on valiaton data
                    if (cur_step+1) % 10 == 0:
                        val_generator = source.valid_generator(args.batch_size)
                        #jut try out one batch
                        x, y  = next(val_generator)
                        feed = {net.image_input:  x,
                            net.labels:           y,
                            net.keep_prob:    1.0,
                            net.is_training: False}
                    
                        sess.run(net.reset_iou_op)
                        summary, _, loss_batch,label_mapper, img_classes = sess.run([net.merged, net.update_iou_op, 
                                                                                        net.loss, net.label_mapper, net.classes], feed_dict=feed)
                        net.val_writer.add_summary(summary, cur_step)
                        iou, summary = sess.run([net.metric_iou__op, net.merged_update])
                        net.val_writer.add_summary(summary, cur_step)
                        print("#####validation: step {}/{}: loss={}, iou={}#####".format(cur_step, step_num, loss_batch, iou))
                        
                        val_imgs = x[:1,:,:,:]
                        val_img_labels = img_classes[:1, :, :]
                        val_img_labels_gt = label_mapper[:1, :, :]
                        imgs_inferred = utils.draw_labels_batch(val_imgs, val_img_labels, source.label_colors)
                        imgs_gt       = utils.draw_labels_batch(val_imgs, val_img_labels_gt, source.label_colors)
                        val_imgs = utils.convert_rgb_batch(val_imgs)
                        all_imgs = np.concatenate([val_imgs, imgs_gt, imgs_inferred], axis = 0)

                        summary = sess.run(validation_img_summary_op,
                                                            feed_dict={validation_imgs: all_imgs})        
                        net.val_writer.add_summary(summary, cur_step)
                    
                if (e+1) % args.checkpoint_interval == 0:
                    checkpoint = '{}/e{}.ckpt'.format(args.name, e+1)
                    saver.save(sess, checkpoint)
                    print('Checkpoint saved:', checkpoint)
                            
                  
        
               
        return
    
if __name__ == "__main__":   
    obj= TrainModel()
    obj.run()
    
