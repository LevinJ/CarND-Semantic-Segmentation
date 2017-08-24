import tensorflow as tf


def metrics_f1(ground_truth, prediction):
    tp = tf.logical_and( tf.equal(ground_truth, True ), tf.equal(prediction, True ))
    tp = tf.reduce_sum(tf.cast(tp, tf.float32))
    
    fp = tf.logical_and( tf.equal(ground_truth, False ), tf.equal(prediction, True ))
    fp = tf.reduce_sum(tf.cast(fp, tf.float32))
    
    fn = tf.logical_and( tf.equal(ground_truth, True ), tf.equal(prediction, False ))
    fn = tf.reduce_sum(tf.cast(fn, tf.float32))
    
    tn = tf.logical_and( tf.equal(ground_truth, False ), tf.equal(prediction, False ))
    tn = tf.reduce_sum(tf.cast(tn, tf.float32))
    
    def compute_precision(tp, fp, name):
        return tf.where(
            tf.greater(tp + fp, 0),
            tf.div(tp, tp + fp),
            1e-4,
            name)
    precision = compute_precision(tp, fp, 'value')
    def compute_recall(true_p, false_n, name):
        return tf.where(
            tf.greater(true_p + false_n, 0),
            tf.div(true_p, true_p + false_n),
            1e-4,
            name)

    recall = compute_recall(tp, fn, 'value')
    
#     precision = tp/(tp + fp)
#     recall = tp/(tp + fn)
    f1 = (2* precision * recall)/(precision + recall)
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    return tp,fp,fn, precision, recall,f1,accuracy

def metrics_f1_summary(ground_truth, prediction):
    _,_,_, precision, recall,f1,accuracy = metrics_f1(ground_truth, prediction)
#     tp_summary = tf.summary.scalar('tp', tp)
#     fp_summary = tf.summary.scalar('fp', fp)
#     fn_summary = tf.summary.scalar('fn', fn)
    precision_summary = tf.summary.scalar('precision', precision)
    recall_summary = tf.summary.scalar('recall', recall)
    f1_summary = tf.summary.scalar('f1', f1)
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    return precision_summary, recall_summary, f1_summary,accuracy_summary, f1,accuracy