import tensorflow as tf
import numpy as np


def mean_iou(ground_truth, prediction, num_classes):
#     iou, iou_op = tf.metrics.mean_iou(ground_truth, prediction, num_classes)
    with tf.variable_scope("resetable_mean_iou") as scope:
        metric_op, update_op = tf.metrics.mean_iou(ground_truth, prediction, num_classes)
        metrics_vars = tf.contrib.framework.get_variables(
                     scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(metrics_vars)
    return metric_op, update_op, reset_op


ground_truth = tf.constant([
    [0, 0, 0, 0], 
    [1, 1, 1, 1], 
    [2, 2, 2, 2], 
    [3, 3, 3, 3]], dtype=tf.float32)
prediction_1 = np.array([
    [0, 0, 0, 0], 
    [1, 0, 0, 1], 
    [1, 2, 2, 1], 
    [3, 3, 0, 3]], dtype=np.float32)


prediction_2 = np.array([
    [0, 0, 0, 0], 
    [1, 1, 1, 1], 
    [2, 2, 2, 2], 
    [3, 3, 3, 3]], dtype=np.float32)


prediction = tf.placeholder(tf.float32, ground_truth.get_shape().as_list())
    
# TODO: use `mean_iou` to compute the mean IoU
iou, iou_op, reset_op = mean_iou(ground_truth, prediction, 4)

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # need to initialize local variables for this to run `tf.metrics.mean_iou`
        sess.run(tf.local_variables_initializer())
        
        print(sess.run(iou_op, feed_dict = {prediction: prediction_1}))
        # should be 0.53869
        print("Mean IoU ={}", sess.run(iou))
        
        sess.run(reset_op)
        
        print(sess.run(iou_op, feed_dict = {prediction: prediction_2}))
        # should be 0.53869
        print("Mean IoU ={}", sess.run(iou))
        
        
        
        
        
        