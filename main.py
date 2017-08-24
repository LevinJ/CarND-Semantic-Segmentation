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
        
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    Input = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', strides=1,kernel_initializer=tf.truncated_normal_initializer(stddev = 1e-3))
        
    pool_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same', strides=1,kernel_initializer=tf.truncated_normal_initializer(stddev = 1e-3))
   
    pool_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same', strides=1,kernel_initializer=tf.truncated_normal_initializer(stddev = 1e-3))
   
    
    #upsample by 2 so that it can match with pool_4
    Input = tf.layers.conv2d_transpose(Input, num_classes, 4, padding='same', strides=2,kernel_initializer=tf.truncated_normal_initializer(stddev = 1e-3))
    Input = tf.add(Input, pool_4)
   
    #upsample by 2 so that it can match with pool_3
    Input = tf.layers.conv2d_transpose(Input, num_classes, 4, padding='same', strides=2,kernel_initializer=tf.truncated_normal_initializer(stddev = 1e-3))
    Input = tf.add(Input, pool_3)
    
    Input = tf.layers.conv2d_transpose(Input, num_classes, 16, padding='same', strides=8,
                                                     kernel_initializer=tf.truncated_normal_initializer(stddev = 1e-3), name="logits_out")
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


def initialize_uninitialized_variables(sess):
    """
    Only initialize the weights that have not yet been initialized by other
    means, such as importing a metagraph and a checkpoint. It's useful when
    extending an existing model.
    """
    uninit_vars    = []
    var_list = tf.global_variables() + tf.local_variables()
    uninit_tensors = []
    for var in var_list:
        uninit_vars.append(var)
        uninit_tensors.append(tf.is_variable_initialized(var))
    uninit_bools = sess.run(uninit_tensors)
    uninit = zip(uninit_bools, uninit_vars)
    uninit = [var for init, var in uninit if not init]
    sess.run(tf.variables_initializer(uninit))

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
    

    initialize_uninitialized_variables(sess)
    for i in range(epochs):
        iter_num = 0
        for images, labels in get_batches_fn(batch_size): 
            iter_num = iter_num + 1

            _, loss= sess.run([train_op, cross_entropy_loss],
                feed_dict={input_image: images, correct_label: labels, keep_prob: 1.0, learning_rate:1e-4})
            
            print("Epoch {}/{}, Loss {:.5f}".format(i, iter_num, loss))
            
                
tests.test_train_nn(train_nn)


def run():

    num_classes = 2
    image_shape = (160, 576)
    epochs = 30
    batch_size = 20
    correct_label = tf.placeholder(tf.float32, shape=[None, image_shape[0],image_shape[1], 2])
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

       
        # Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        preidction_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(preidction_layer, correct_label, learning_rate, num_classes)
       
        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
             correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)
        print("Done")

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
