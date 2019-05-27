import tensorflow as tf
import os
import numpy as np
import h5py
from scipy.misc import imsave
import tensorlayer as tl

"""This script is used for data processing"""
def savemat2tfrecord(tf_record_name, mat_file, crop_number, crop_size, NUM_CHANNELS, W_crop,Shuffle = False):
    """Function interface for savemat2tfrecord
        This function helps address the matdata with different data augmentation, and save as tfrecord type
        Arguments:
            tf_record_name: the name of saved tfrecord_file
            mat_file: the filepath of .mat to read
            crop_number: the number of crop
            crop_size: crop size of final images
            NUM_CHANNELS: the channel of input
            W_crop: Whether crop or not
    """
    crop_number = crop_number
    crop_patch_size = crop_size
    NUM_CHANNELS = NUM_CHANNELS
    tfrecord_filename = tf_record_name
    Path = mat_file

    PE_size_ori = 288
    FE_size_ori = 384
    #
    # these two array is prepared for crop
    batch_x= np.zeros((1,FE_size_ori,PE_size_ori,NUM_CHANNELS)).astype('float32')
    batch_y = np.zeros((1,FE_size_ori,PE_size_ori,NUM_CHANNELS)).astype('float32')
    # # end

    # bulid the tfrecord writer
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    # # end

    # # read the mat data
    s1 = h5py.File(Path  + '/low_CompI_final.mat')
    X_data = s1[('low_CompI_final')]
    s1 = h5py.File(Path  + '/CompI_final.mat')
    Y_data = s1[('CompI_final')]
    if NUM_CHANNELS == 1:
        X_data = np.transpose(X_data, [2, 1, 0])
        Y_data = np.transpose(Y_data, [2, 1, 0])
    else:
        X_data = np.transpose(X_data,[3, 2, 1, 0])
        Y_data = np.transpose(Y_data, [3, 2, 1, 0])
        X_data = np.concatenate((X_data[:,:,:,:3],X_data[:,:,:,3:]),axis= 2)
        Y_data = np.concatenate((Y_data[:, :, :, :3], Y_data[:, :, :, 3:]), axis=2)

    # # end

    nb_images = Y_data.shape[2]
    indices = np.arange(nb_images)

    if Shuffle == True:
        np.random.shuffle(indices)
    for j in indices:
        for i in range(crop_number):
            #
            for k in range(NUM_CHANNELS):
                if NUM_CHANNELS == 1:
                    batch_y[0, :, :, 0] = Y_data[:, :, j].astype('float32')
                    batch_x[0, :, :, 0] = X_data[:, :, j].astype('float32')
                else:
                    batch_y[0, :, :, k] = Y_data[:, :, j, k].astype('float32')
                    batch_x[0, :, :, k] = X_data[:, :, j, k].astype('float32')

            Input = np.concatenate((batch_x, batch_y))
            # ##### crop
            if W_crop == True:
                Output = tl.prepro.crop_multi(Input,crop_patch_size,crop_patch_size,True).astype('float32')
            else:
                Output = Input
            # Output = tl.prepro.flip_axis_multi(Output, axis=1, is_random=True)
            # Output = tl.prepro.rotation_multi(Output,rg=10, is_random=True, fill_mode='constant')
            # Output = tl.prepro.shift_multi(Output, wrg=0.10, hrg=0.10, is_random=True,fill_mode='constant')
            # Output = tl.prepro.zoom_multi(Output,zoom_range=[0.90,1.10],is_random=True, fill_mode='constant')
            # Output = tl.prepro.brightness_multi(Output,gamma=0.05, is_random=True)
            X_small_patch = Output[0,:,:,0:NUM_CHANNELS].tostring()
            Y_small_patch = Output[1,:,:,0:NUM_CHANNELS].tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'low_CompI': tf.train.Feature(bytes_list=tf.train.BytesList(value=[X_small_patch])),
                             'CompI': tf.train.Feature(bytes_list=tf.train.BytesList(value=[Y_small_patch]))
                             }
            )
            )
            # X_small_patch[k:,:,0],Y_small_patch[k,:,:,0]  = tl.prepro.crop_multi([X_input, Y_input],16,16,True)
            serialized = example.SerializeToString()
            writer.write(serialized)
    writer.close()

def tfrecord_read_dataset(batch_size, size_FE, size_PE, Filenames, c_dim, training):
    """
    read binary data from filenames
    Argument:
        batch_size:
        size_FE: FE Num of data
        size_PE: PE Num of data
        Filenames: name of tfrecord
        c_dim: channel
        training: whether shuffle or not
    return: batch_x, batch_y
    """
    def parser(record):
        features = tf.parse_single_example(record,
                                           features={
                                               'low_CompI': tf.FixedLenFeature([], tf.string),
                                               'CompI': tf.FixedLenFeature([], tf.string)
                                           })
        low = tf.decode_raw(features['low_CompI'], tf.float32)
        low = tf.reshape(low, [crop_patch_FE, crop_patch_PE,Num_CHANNELS])

        high = tf.decode_raw(features['CompI'], tf.float32)
        high = tf.reshape(high, [crop_patch_FE, crop_patch_PE,Num_CHANNELS])
        return low, high
    crop_patch_FE = size_FE
    crop_patch_PE = size_PE
    Num_CHANNELS = c_dim
    batch_size = batch_size
    if training == True:
        buffer_size = 20000
    else:
        buffer_size = 1
    # output file name string to a queue
    dataset = tf.data.TFRecordDataset(Filenames)
    dataset = dataset.map(parser)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size= buffer_size).batch(batch_size)
    itertor = dataset.make_one_shot_iterator()
    final = itertor.get_next()
    return final

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


def data_inputs(testing_filename,testing_FE,testing_PE, channel, shuffle=False, Batch_size=1):
    """Function interface for data_input
        This function helps read the matdata for training or testing
        Arguments:
            config: config for parameters
            channel: number channel of input and output
            testing_filename: the filepath of .mat to read, including low_CompI_final, CompI_final
            shuffle: whether random
            Batch_size:
    """

    image_shape = (testing_FE, testing_PE, channel)
    y_image_shape = (testing_FE, testing_PE, channel)
    NUM_CHANNELS = channel

    s1 = h5py.File(testing_filename + '\low_CompI_final.mat')
    X_data = s1['low_CompI_final'].value
    s2 = h5py.File(testing_filename + '\CompI_final.mat')
    Y_data = s2['CompI_final'].value

    # for s in range(10):
    if NUM_CHANNELS == 1:
        X_data = np.transpose(X_data, [2, 1, 0])
        Y_data = np.transpose(Y_data, [2, 1, 0])
    else:
        X_data = np.transpose(X_data, [3, 2, 1, 0])
        Y_data = np.transpose(Y_data, [3, 2, 1, 0])
    nb_images = Y_data.shape[2]
    # nb_images =10
    index_generator = _index_generator(nb_images, Batch_size, shuffle, None)

    while 1:
        index_array, current_index, current_batch_size = next(index_generator)

        batch_x = np.zeros((current_batch_size,) + image_shape)

        batch_y = np.zeros((current_batch_size,) + y_image_shape)

        for i, j in enumerate(index_array):

            for k in range(NUM_CHANNELS):
                if NUM_CHANNELS == 1:
                    batch_y[0, :, :, 0] = Y_data[:, :, j].astype('float32')
                    batch_x[0, :, :, 0] = X_data[:, :, j].astype('float32')
                else:
                    batch_y[0, :, :, k] = Y_data[:, :, j, k].astype('float32')
                    batch_x[0, :, :, k] = X_data[:, :, j, k].astype('float32')

        yield (batch_x, batch_y)

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

def main(_):
    tfrecord_filename = 'Amp_3channel_xuwei_qin_l_0_01.tfrecord'
    Path = 'F:\matlab\Data_address_cml\BrainQuant_AI\Qin_Data\XUWEI\\6channel'
    ##### for training
    # savemat2tfrecord(tfrecord_filename,Path,50,80,3,True,True)
    ##### for testing
    savemat2tfrecord(tfrecord_filename,Path,1,80,3,False,False)
    # datasetimg=tfrecord_read_dataset(1,80,80,'Amp_one_channel_13vol.tfrecord',1, True)
    # with tf.Session() as sess:
    #     imx, imy = sess.run(datasetimg)
    #     print(imx.shape)
if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ' 1'
    tf.app.run()


