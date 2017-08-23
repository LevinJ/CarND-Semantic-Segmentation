import tensorflow as tf
import numpy as np



ground_truth = tf.constant([
    [0, 0, 0, 0], 
    [1, 1, 1, 1]], dtype=tf.float32)
prediction_1 = np.array([
    [0, 0, 0, 0], 
    [1, 0, 0, 1]], dtype=np.float32)


prediction_2 = np.array([
    [0, 0, 0, 0], 
    [1, 1, 1, 1]], dtype=np.float32)


prediction = tf.placeholder(tf.float32, ground_truth.get_shape().as_list())
    

def metrics_f1(ground_truth, prediction):
    tp = tf.logical_and( tf.equal(ground_truth, True ), tf.equal(prediction, True ))
    tp = tf.reduce_sum(tf.cast(tp, tf.float32))
    
    fp = tf.logical_and( tf.equal(ground_truth, False ), tf.equal(prediction, True ))
    fp = tf.reduce_sum(tf.cast(fp, tf.float32))
    
    fn = tf.logical_and( tf.equal(ground_truth, True ), tf.equal(prediction, False ))
    fn = tf.reduce_sum(tf.cast(fn, tf.float32))
    
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = (2* precision * recall)/(precision + recall)
    return tp,fp,fn, precision, recall,f1

tp,fp,fn, precision, recall,f1 = metrics_f1(ground_truth, prediction)



with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # need to initialize local variables for this to run `tf.metrics.mean_iou`
        sess.run(tf.local_variables_initializer())
        
        res = sess.run([tp,fp,fn, precision, recall,f1], feed_dict = {prediction: prediction_1})
        print(res)
        
       
        
        
        
        
        