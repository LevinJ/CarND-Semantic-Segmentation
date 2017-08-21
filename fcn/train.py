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
parser.add_argument('--epochs', type=int, default=10,
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
    def __load_model(self):
        return
    def __train(self, sess, source, net,e):
        n_train_batches = int(math.ceil(source.num_training/args.batch_size))
        trainmetrics = APMetrics()
        generator = source.train_generator(args.batch_size)
        description = '[i] Epoch {:>2}/{}'.format(e+1, args.epochs)
        training_loss_total = 0
        for x, y in tqdm(generator, total=n_train_batches,
                         desc=description, unit='batches'):
            feed = {net.image_input:  x,
                    net.labels:           y,
                    net.keep_prob:    0.5}
            loss_batch, _,y_softmax = sess.run([net.loss, net.optimizer,net.y_softmax], feed_dict=feed)
            trainmetrics.y_true.append(y)
            trainmetrics.y_score.append(y_softmax)
            training_loss_total += loss_batch * x.shape[0]
        training_loss_total /= source.num_training
        self.training_loss_total = training_loss_total
        print("train ap={}".format(trainmetrics.get_ap()))
        return
    def __validate(self, sess, source, net, e):
        valmetrics = APMetrics()
        generator = source.valid_generator(args.batch_size)
        validation_loss_total = 0
        self.val_imgs          = None
        self.val_img_labels    = None
        self.val_img_labels_gt = None
        for x, y in generator:
            feed = {net.image_input:  x,
                    net.labels:           y,
                    net.keep_prob:    1}
            loss_batch, img_classes, y_mapped, y_softmax = sess.run([net.loss,
                                                          net.classes,
                                                          net.label_mapper,
                                                          net.y_softmax],
                                                         feed_dict=feed)
            valmetrics.y_true.append(y)
            valmetrics.y_score.append(y_softmax)
            validation_loss_total += loss_batch * x.shape[0]

            if self.val_imgs is None:
                self.val_imgs          = x[:3, :, :, :]
                self.val_img_labels    = img_classes[:3, :, :]
                self.val_img_labels_gt = y_mapped[:3, :, :]

        validation_loss_total /= source.num_validation
        self.validation_loss_total = validation_loss_total
        
        print("val ap={}".format(valmetrics.get_ap()))
        
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
        
            summary_writer  = tf.summary.FileWriter(args.tensorboard_dir, sess.graph)
            saver           = tf.train.Saver(max_to_keep=10)
        
            
            
        
            utils.initialize_uninitialized_variables(sess)
            print('[i] Training...')
        
            #---------------------------------------------------------------------------
            # set up summaries
            #---------------------------------------------------------------------------
            validation_loss = tf.placeholder(tf.float32)
            validation_loss_summary_op = tf.summary.scalar('validation_loss',
                                                           validation_loss)
        
            training_loss = tf.placeholder(tf.float32)
            training_loss_summary_op = tf.summary.scalar('training_loss',
                                                         training_loss)
        
            validation_img    = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            validation_img_gt = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            validation_img_summary_op = tf.summary.image('validation_img',
                                                         validation_img)
            validation_img_gt_summary_op = tf.summary.image('validation_img_gt',
                                                            validation_img_gt)
            validation_img_summary_ops = [validation_img_summary_op,
                                          validation_img_gt_summary_op]
        
            for e in range(args.epochs):
                #-----------------------------------------------------------------------
                # Train
                #-----------------------------------------------------------------------
                self.__train(sess, source, net,e)
        
                #-----------------------------------------------------------------------
                # Validate
                #-----------------------------------------------------------------------
                
                self.__validate(sess, source, net, e)
                #-----------------------------------------------------------------------
                # Write  summary
                #-----------------------------------------------------------------------
                feed = {validation_loss: self.validation_loss_total,
                        training_loss:   self.training_loss_total}
                loss_summary = sess.run([validation_loss_summary_op,
                                         training_loss_summary_op],
                                        feed_dict=feed)
        
                summary_writer.add_summary(loss_summary[0], e)
                summary_writer.add_summary(loss_summary[1], e)
                
                if e % 1 == 0:
                    imgs_inferred = utils.draw_labels_batch(self.val_imgs, self.val_img_labels, source.label_colors)
                    imgs_gt       = utils.draw_labels_batch(self.val_imgs, self.val_img_labels_gt, source.label_colors)
        
                    feed = {validation_img:    imgs_inferred,
                            validation_img_gt: imgs_gt}
                    validation_img_summaries = sess.run(validation_img_summary_ops,
                                                        feed_dict=feed)
                    summary_writer.add_summary(validation_img_summaries[0], e)
                    summary_writer.add_summary(validation_img_summaries[1], e)
        
                
        
                #-----------------------------------------------------------------------
                # Save a checktpoint
                #-----------------------------------------------------------------------
                if (e+1) % args.checkpoint_interval == 0:
                    checkpoint = '{}/e{}.ckpt'.format(args.name, e+1)
                    saver.save(sess, checkpoint)
                    print('Checkpoint saved:', checkpoint)
        
            checkpoint = '{}/final.ckpt'.format(args.name)
            saver.save(sess, checkpoint)
            print('Checkpoint saved:', checkpoint)
        return
    
if __name__ == "__main__":   
    obj= TrainModel()
    obj.run()
    
