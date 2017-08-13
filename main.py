import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests



# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    #   Use tf.saved_model.loader.load to load the model and weights

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    vgg_graph =  tf.get_default_graph()
    
    image_input = vgg_graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = vgg_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = vgg_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = vgg_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = vgg_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
        
# tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    ###May need to come back and check the weight initializer, for now, just use the default one###
    #complete the encoder 
    Input = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', strides=1)
    
    #complete the decoder
    pool_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same', strides=1)
    pool_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same', strides=1)
    
    #upsample by 2 so that it can match with pool_4
    Input = tf.layers.conv2d_transpose(Input, num_classes, 4, padding='same', strides=2)
    Input = tf.add(Input, pool_4)
    #upsample by 2 so that it can match with pool_3
    Input = tf.layers.conv2d_transpose(Input, num_classes, 4, padding='same', strides=2)
    Input = tf.add(Input, pool_3)
    
    #upsample by 5 so that it will be the same size as the orginal image
    Input = tf.layers.conv2d_transpose(Input, num_classes, 16, padding='same', strides=8)
    return Input
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)

g_iou = None
g_iou_op = None
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    

    train_writer = tf.summary.FileWriter('./logs', sess.graph)
    merge = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    # need to initialize local variables for this to run `tf.metrics.mean_iou`
    sess.run(tf.local_variables_initializer())

    for i in range(epochs):
        iter_num = 0
        for images, labels in get_batches_fn(batch_size): 
#             break
            iter_num = iter_num + 1
            if images.shape[0] != batch_size:
                continue

            _, loss,_, iou= sess.run([train_op, cross_entropy_loss,g_iou_op,g_iou],
                feed_dict={input_image: images, correct_label: labels, keep_prob: 1.0, learning_rate:1e-3})
            
#             train_writer.add_summary(summary, i)

            print("Epoch {}/{}, Loss {:.5f}, IOU {}".format(i, iter_num, loss, iou))
            
                
# tests.test_train_nn(train_nn)


def run():
    global g_iou
    global g_iou_op
    num_classes = 2
    image_shape = (160, 576)
    epochs = 1
    batch_size = 20
    correct_label = tf.placeholder(tf.float32, shape=[batch_size, image_shape[0],image_shape[1], 2])
    learning_rate = tf.placeholder(tf.float32, shape=[])

    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
       
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        preidction_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(preidction_layer, correct_label, learning_rate, num_classes)
        
        g_iou, g_iou_op = tf.metrics.mean_iou(tf.argmax(tf.reshape(correct_label, (-1,2)), -1), tf.argmax(logits, -1), num_classes)
        
        
       
        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
             correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
#         helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
