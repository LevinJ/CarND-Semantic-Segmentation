import zipfile
import shutil
import os

import tensorflow as tf

from urllib.request import urlretrieve
from upscale import upsample
from tqdm import tqdm
import numpy as np
import f1metrics

#-------------------------------------------------------------------------------
class DLProgress(tqdm):
    last_block = 0

    #---------------------------------------------------------------------------
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

#-------------------------------------------------------------------------------
def reshape(x, num_classes, upscale_factor, name):
    """
    Reshape the tensor so that it matches the number of classes and output size
    :param x:              input tensor
    :param num_classes:    number of classes
    :param upscale_factor: scaling factor
    :param name:           name of the resulting tensor
    :return:               reshaped tensor
    """
    with tf.variable_scope(name):
        w_shape = [1, 1, int(x.get_shape()[3]), num_classes]
        w = tf.Variable(tf.truncated_normal(w_shape, 0, 0.1),
                        name=name+'_weights')
        b = tf.Variable(tf.zeros(num_classes), name=name+'_bias')
        resized = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID',
                               name=name+'_resized')
        resized = tf.nn.bias_add(resized, b, name=name+'_add_bias')

    upsampled = upsample(resized, num_classes, upscale_factor,
                         name+'_upsampled')
    return upsampled

#-------------------------------------------------------------------------------
class FCNVGG:
    #---------------------------------------------------------------------------
    def __init__(self, session, num_classes):
        self.session     = session
        self.labels = tf.placeholder(tf.float32,
                                    shape=[None, None, None, num_classes])
        self.label_mapper    = tf.argmax(self.labels, axis=3)
        self.is_training = tf.placeholder(tf.bool, name="is_training_placeholder")

    #---------------------------------------------------------------------------
    def build_from_vgg(self, vgg_dir, num_classes, progress_hook):
        """
        Build the model for training based on a pre-define vgg16 model.
        :param vgg_dir:       directory where the vgg model should be stored
        :param num_classes:   number of classes
        :param progress_hook: a hook to show download progress of vgg16;
                              the value may be a callable for urlretrieve
                              or string "tqdm"
        """
        self.num_classes = num_classes
        self.__download_vgg(vgg_dir, progress_hook)
        self.__load_vgg(vgg_dir)
        self.__make_result_tensors()

    #---------------------------------------------------------------------------
    def build_from_metagraph(self, metagraph_file, checkpoint_file):
        """
        Build the model for inference from a metagraph shapshot and weights
        checkpoint.
        """
        sess = self.session
        saver = tf.train.import_meta_graph(metagraph_file)
        saver.restore(sess, checkpoint_file)
#         all_tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]
#         print(all_tensors)
        self.is_training = sess.graph.get_tensor_by_name('is_training_placeholder_1:0')
        self.image_input = sess.graph.get_tensor_by_name('image_input:0')
        self.keep_prob   = sess.graph.get_tensor_by_name('keep_prob:0')
        self.logits      = sess.graph.get_tensor_by_name('logits/logits_out/conv2d_transpose:0')
        self.softmax     = sess.graph.get_tensor_by_name('result/Softmax:0')
        self.classes     = sess.graph.get_tensor_by_name('result/ArgMax:0')

    #---------------------------------------------------------------------------
    def __download_vgg(self, vgg_dir, progress_hook):
        #-----------------------------------------------------------------------
        # Check if the model needs to be downloaded
        #-----------------------------------------------------------------------
        vgg_archive = 'vgg.zip'
        vgg_files   = [
            vgg_dir + '/variables/variables.data-00000-of-00001',
            vgg_dir + '/variables/variables.index',
            vgg_dir + '/saved_model.pb']

        missing_vgg_files = [vgg_file for vgg_file in vgg_files \
                             if not os.path.exists(vgg_file)]

        if missing_vgg_files:
            if os.path.exists(vgg_dir):
                shutil.rmtree(vgg_dir)
            os.makedirs(vgg_dir)

            #-------------------------------------------------------------------
            # Download vgg
            #-------------------------------------------------------------------
            url = 'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip'
            if not os.path.exists(vgg_archive):
                if callable(progress_hook):
                    urlretrieve(url, vgg_archive, progress_hook)
                else:
                    with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
                        urlretrieve(url, vgg_archive, pbar.hook)

            #-------------------------------------------------------------------
            # Extract vgg
            #-------------------------------------------------------------------
            zip_archive = zipfile.ZipFile(vgg_archive, 'r')
            zip_archive.extractall(vgg_dir)
            zip_archive.close()

    #---------------------------------------------------------------------------
    def __load_vgg(self, vgg_dir):
        sess = self.session
        graph = tf.saved_model.loader.load(sess, ['vgg16'], vgg_dir)
#         all_tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]
#         for n in tf.get_default_graph().as_graph_def().node:
#             if n.name == 'fc6/Conv2D':
#                 print(n)
                
#         print(np.array(all_tensors))
        self.image_input = sess.graph.get_tensor_by_name('image_input:0')
        self.keep_prob   = sess.graph.get_tensor_by_name('keep_prob:0')
        self.vgg_layer3  = sess.graph.get_tensor_by_name('layer3_out:0')
        self.vgg_layer4  = sess.graph.get_tensor_by_name('layer4_out:0')
#         self.vgg_pool5  = sess.graph.get_tensor_by_name('pool5:0')
#         self.vgg_layer6  = sess.graph.get_tensor_by_name('fc6/Relu:0')
        self.vgg_layer7  = sess.graph.get_tensor_by_name('layer7_out:0')

    #---------------------------------------------------------------------------
    def __make_result_tensors(self):
#         vgg3_reshaped = reshape(self.vgg_layer3, self.num_classes,  8,
#                                 'layer3_resize')
#         vgg4_reshaped = reshape(self.vgg_layer4, self.num_classes, 16,
#                                 'layer4_resize')
#         vgg7_reshaped = reshape(self.vgg_layer7, self.num_classes, 32,
#                                 'layer7_resize')
# 
#         with tf.variable_scope('sum'):
#             self.logits   = tf.add(vgg3_reshaped,
#                                    tf.add(2*vgg4_reshaped, 4*vgg7_reshaped))
       
#         Input = tf.layers.conv2d(self.vgg_layer7, self.num_classes, 1, padding='same', strides=1,kernel_initializer=tf.contrib.layers.xavier_initializer())
#         Input = tf.layers.batch_normalization(Input, training=self.is_training)
#         pool_3 = tf.layers.conv2d(self.vgg_layer3, self.num_classes, 1, padding='same', strides=1,kernel_initializer=tf.contrib.layers.xavier_initializer())
#         pool_3 = tf.layers.batch_normalization(pool_3, training=self.is_training)
#         pool_4 = tf.layers.conv2d(self.vgg_layer4, self.num_classes, 1, padding='same', strides=1,kernel_initializer=tf.contrib.layers.xavier_initializer())
#         pool_4 = tf.layers.batch_normalization(pool_4, training=self.is_training)
#         
#         #upsample by 2 so that it can match with pool_4
#         Input = tf.layers.conv2d_transpose(Input, self.num_classes, 4, padding='same', strides=2,kernel_initializer=tf.contrib.layers.xavier_initializer())
#         Input = tf.add(Input, pool_4)
#         Input = tf.layers.batch_normalization(Input, training=self.is_training)
#         #upsample by 2 so that it can match with pool_3
#         Input = tf.layers.conv2d_transpose(Input, self.num_classes, 4, padding='same', strides=2,kernel_initializer=tf.contrib.layers.xavier_initializer())
#         Input = tf.add(Input, pool_3)
#         Input = tf.layers.batch_normalization(Input, training=self.is_training)
#         
#         #upsample by 5 so that it will be the same size as the orginal image
#         with tf.variable_scope('logits'):
#             self.logits = tf.layers.conv2d_transpose(Input, self.num_classes, 16, padding='same', strides=8,
#                                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), name="logits_out")
#                 
#         with tf.name_scope('result'):
#             self.y_softmax  = tf.nn.softmax(self.logits)
#             self.classes  = tf.argmax(self.y_softmax, axis=3)
        
        
        Input = tf.layers.conv2d(self.vgg_layer7, self.num_classes, 1, padding='same', strides=1,kernel_initializer=tf.truncated_normal_initializer(stddev = 1e-3))
        
        pool_3 = tf.layers.conv2d(self.vgg_layer3, self.num_classes, 1, padding='same', strides=1,kernel_initializer=tf.truncated_normal_initializer(stddev = 1e-3))
       
        pool_4 = tf.layers.conv2d(self.vgg_layer4, self.num_classes, 1, padding='same', strides=1,kernel_initializer=tf.truncated_normal_initializer(stddev = 1e-3))
       
        
        #upsample by 2 so that it can match with pool_4
        Input = tf.layers.conv2d_transpose(Input, self.num_classes, 4, padding='same', strides=2,kernel_initializer=tf.truncated_normal_initializer(stddev = 1e-3))
        Input = tf.add(Input, pool_4)
       
        #upsample by 2 so that it can match with pool_3
        Input = tf.layers.conv2d_transpose(Input, self.num_classes, 4, padding='same', strides=2,kernel_initializer=tf.truncated_normal_initializer(stddev = 1e-3))
        Input = tf.add(Input, pool_3)
        
        
        #upsample by 5 so that it will be the same size as the orginal image
        with tf.variable_scope('logits'):
            self.logits = tf.layers.conv2d_transpose(Input, self.num_classes, 16, padding='same', strides=8,
                                                     kernel_initializer=tf.truncated_normal_initializer(stddev = 1e-3), name="logits_out")
                
        with tf.name_scope('result'):
            self.y_softmax  = tf.nn.softmax(self.logits)
            self.classes  = tf.argmax(self.y_softmax, axis=3)
            
            
    def mean_iou(self, ground_truth, prediction, num_classes):
        with tf.variable_scope("resetable_mean_iou") as scope:
            metric_op, update_op = tf.metrics.mean_iou(ground_truth, prediction, num_classes)
            metrics_vars = tf.contrib.framework.get_variables(
                         scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
            reset_op = tf.variables_initializer(metrics_vars)
        return metric_op, update_op, reset_op
    def add_summary_nodes(self, summaries_dir):
        
        self.metric_iou__op, self.update_iou_op, self.reset_iou_op = self.mean_iou(self.label_mapper, self.classes, self.num_classes)
        
        self.merged_update = tf.summary.merge([tf.summary.scalar('iou', self.metric_iou__op)])
        
        precision_summary, recall_summary, f1_summary,accuracy_summary, self.f1,self.accuracy = f1metrics.metrics_f1_summary(self.label_mapper, self.classes)
        loss_summary = tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge([loss_summary, precision_summary, recall_summary, f1_summary,accuracy_summary])
        self.train_writer = tf.summary.FileWriter(summaries_dir+ '/train',
                                        self.session.graph)
        self.val_writer = tf.summary.FileWriter(summaries_dir + '/val')
        return

    #---------------------------------------------------------------------------
    def get_optimizer(self, labels, learning_rate=0.0001):
        with tf.variable_scope('reshape'):
            labels_reshaped  = tf.reshape(labels, [-1, self.num_classes])
            logits_reshaped  = tf.reshape(self.logits, [-1, self.num_classes])
            losses          = tf.nn.softmax_cross_entropy_with_logits(
                                  labels=labels_reshaped,
                                  logits=logits_reshaped)
            loss            = tf.reduce_mean(losses)
        with tf.variable_scope('optimizer'):
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                optimizer       = tf.train.AdamOptimizer(learning_rate)
                optimizer       = optimizer.minimize(loss)
        self.optimizer = optimizer
        self.loss = loss

        return
