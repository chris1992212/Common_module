import tensorflow as tf
import numpy as np
from time import localtime, strftime
import os
import logging
import json
import tensorlayer as tl
from tensorlayer.layers import *

def _index_generator(N, batch_size, shuffle=True, seed=None):
    batch_index = 0
    total_batches_seen = 0

    while 1:
        if seed is not None:
            np.random.seed(seed + total_batches_seen)

        current_index = (batch_index * batch_size) % N
        if current_index == 0:
            index_array = np.arange(N)
            if shuffle:
                index_array = np.random.permutation(N)
        if N >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            # current_batch_size = N - current_index
            current_batch_size = batch_size
            batch_index = 0
            current_index = 0
            if shuffle:
                index_array = np.random.permutation(N)
        total_batches_seen += 1

        yield (index_array[current_index: current_index + current_batch_size],
               current_index, current_batch_size)
def tfrecord_read(batch_size, size_FE, size_PE, Filenames, c_dim, training):

    crop_patch_FE = size_FE
    crop_patch_PE = size_PE
    Num_CHANNELS = c_dim
    batch_size = batch_size
    # output file name string to a queue
    filename_queue = tf.train.string_input_producer([Filenames], num_epochs=None)
    # create a reader from file queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'low_CompI': tf.FixedLenFeature([], tf.string),
                                           'CompI': tf.FixedLenFeature([], tf.string)
                                       })
    low = tf.decode_raw(features['low_CompI'], tf.float32)
    low = tf.reshape(low, [crop_patch_FE, crop_patch_PE,Num_CHANNELS])

    high = tf.decode_raw(features['CompI'], tf.float32)
    high = tf.reshape(high, [crop_patch_FE, crop_patch_PE,Num_CHANNELS])
    # ###for three echo
    # low = tf.decode_raw(features['low_CompI'], tf.float64)
    # low = tf.reshape(low, [crop_patch_FE, crop_patch_PE,6])
    # low = low[:,:,3:6]
    #
    # high = tf.decode_raw(features['CompI'], tf.float64)
    # high = tf.reshape(high, [crop_patch_FE, crop_patch_PE,6])
    # high = high[:,:,3:6]
    ####
    if crop_patch_FE > 200:
        capcity = 100
        min_after_dequeue = 10
    else:
        capcity = 20000
        min_after_dequeue = 15000
    if training == True:
        low_batch, high_batch = tf.train.shuffle_batch([low, high], batch_size, capacity=capcity, min_after_dequeue=min_after_dequeue,num_threads=1)
    else:
        low_batch, high_batch = tf.train.batch([low, high], batch_size, capacity=capcity)

    low_image = tf.reshape(low_batch,[batch_size,crop_patch_FE, crop_patch_PE,Num_CHANNELS])
    high_image = tf.reshape(high_batch,[batch_size,crop_patch_FE, crop_patch_PE,Num_CHANNELS])


    return low_image,high_image
def inference_block(input,x,reuse= False,name ='Inference Block' ):
    """Function interface for the paper : A Deep Information Sharing Network ...DISN
        This function bulids the block to make up the whole network
        Arguments:
            input: tensor input from previous block
            x: original input

    """

    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    with tf.variable_scope(name, reuse= reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = tl.layers.InputLayer(input, name='inputs')
        conv1 = tl.layers.Conv2d(inputs, n_filter=32, filter_size = (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv1')

        conv2 = tl.layers.Conv2d(conv1,n_filter=32, filter_size = (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv2')
        conv3 = tl.layers.Conv2d(conv2,n_filter=32, filter_size = (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv3')
        conv4 = tl.layers.Conv2d(conv1, n_filter=6, filter_size=(3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                         b_init=b_init,
                         name='output_for_DC')
        # final_outputs = Data_Consisdency(x,conv4.outputs)
        final_outputs = conv4.outputs
    return  final_outputs
def Concatenation(layers,name):
    return tl.layers.ConcatLayer(layers, concat_dim=3,name=name)
def Data_Consisdency(x, y, ori_size = 324):
    """Function interface for Data Consisdency
        This function bulids the Data Consisdency to pad the original acquisition lines
        Arguments:
            x: original input
            y: images from previous layer
            ori_size: for the real acquisition trajectory
    """
    _,nx,ny,nz = x.get_shape().as_list()

    l = np.int((ori_size - ny)/2)
    x = tf.pad(x,[[0,0],[0,0],[l,l],[0,0]],'constant')
    y = tf.pad(y,[[0,0],[0,0],[l,l],[0,0]],'constant')

    for k in range(nz):
        k_conv3[:,:,:,k] = utils.Fourier(y[:, :, :, k], separate_complex=False)
        mask = np.zeros((1, nx, ori_size))
        mask[:, :, 0:ny:3] = 1
        mask = np.fft.ifftshift(mask)
    # convert to complex tf tensor
        DEFAULT_MAKS_TF = tf.cast(tf.constant(mask), tf.float32)
        DEFAULT_MAKS_TF_c = tf.cast(DEFAULT_MAKS_TF, tf.complex64)
        k_patches = utils.Fourier(x[:, :, :,0], separate_complex=False)
        k_space = k_conv3 * (1 - DEFAULT_MAKS_TF_c) + k_patches * (DEFAULT_MAKS_TF_c)
        out = tf.ifft2d(k_space)
        out = out[:, :, l:ori_size - l]
    # out_real = tf.real(out)
    # out_real = tf.reshape(out_real, [-1, nx, ny, 1])
    # out_imag = tf.imag(out)
    # out_imag = tf.reshape(out_imag, [-1, nx, ny, 1])
        out_abs[:,:,:,k] = tf.abs(out)
        # out_abs = tf.reshape(out_abs, [-1, nx, ny, 1])
    # final_output = tf.concat([out_real, out_imag, out_abs], 3)
    final_output = out_abs
    return final_output
def Fourier(x, separate_complex=True):
    x = tf.cast(x, tf.complex64)
    if separate_complex:
        x_complex = x[:,:,:,0]+1j*x[:,:,:,1]
    else:
        x_complex = x
    # x_complex = tf.reshape(x_complex,x_complex.get_shape()[:3])
    y_complex = tf.fft2d(x_complex)
    print('using Fourier, input dim {0}, output dim {1}'.format(x.get_shape(), y_complex.get_shape()))
    # x = tf.cast(x, tf.complex64)
    # y = tf.fft3d(x)
    # y = y[:,:,:,-1]
    return y_complex

def log_record(config):
    log_dir = "log_{}".format(config.logs_file)
    tl.files.exists_or_mkdir(log_dir)
    log_all, log_all_filename = logging_setup(log_dir)
    log_config(log_all_filename, config)

    ###
def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
def logging_setup(log_dir):
    current_time_str = strftime("%Y_%m_%d_%H", localtime())
    log_all_filename = os.path.join(log_dir, 'log_all_{}.log'.format(current_time_str))
    log_all = logging.getLogger('log_all')
    log_all.setLevel(logging.DEBUG)
    log_all.addHandler(logging.FileHandler(log_all_filename))


    return log_all, log_all_filename