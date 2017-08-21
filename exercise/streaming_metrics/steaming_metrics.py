import tensorflow as tf
import numpy as np

values =  tf.placeholder(tf.float32)
# mean_value, update_op = tf.contrib.metrics.streaming_mean(values, name="train_loss_mean")
# 
real_values = np.arange(5)

train_loss_summary_op = tf.summary.scalar('train_loss', values)
train_loss_summary_op_2 = tf.summary.scalar('train_loss', values)

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # need to initialize local variables for this to run `tf.metrics.mean_iou`
        sess.run(tf.local_variables_initializer())
        
        validation_loss_summary_op = tf.get_collection(tf.GraphKeys.SUMMARIES, scope="train_loss")
        print(validation_loss_summary_op)
        
        
        
        
#         for i in real_values:
#             feed = {values:    i}
#             cur_batch = sess.run(update_op, feed_dict=feed)
#             print('Mean after batch {}: {}'.format(i, cur_batch))
#         print('Final Mean: %f' % mean_value.eval())
#         
#         local_vars = tf.local_variables()
#         for var in local_vars:
#             print(var)
#         validation_loss_summary_op = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="train_loss_mean")
#         sess.run(tf.variables_initializer(train_loss_mean))
#         
#         for i in real_values:
#             feed = {values:    i}
#             cur_batch = sess.run(update_op, feed_dict=feed)
#             print('Mean after batch {}: {}'.format(i, cur_batch))
#         print('Final Mean: %f' % mean_value.eval())
