def model(self, images, is_train=False, reuse=False):
    """
    The network is based on u_net model:
    including two part:
    Downsampling part: 10 convolution layers, 4 max_pooling layers, 4 Batch Normalization layers
    Upsampling part: 8 deconvoluiton layers, 1*1 convolution layers, 4 Batch Normalization layers
    Besides, Concat is used to connect the downsampling and upsampling data

    :param images: input of model, always undersampled images
    :param is_train: to determine the parameters of Batch Normalization
    :param reuse: if 1, is to reuse the name of network layers
    :return:
    """
    n_out = self.c_dim
    x = images
    _, nx, ny, nz = x.get_shape().as_list()

    # ============= To initialize the parameters of network layers ==========#
    # ============ 1. w_int, b_init: initial value of weight and bias of convontion kernels in network====== #
    # ============ 2. gamma_init: initial value of parameters in  Batch normalization algorithm ==== #
    w_init = tf.truncated_normal_initializer(stddev=0.01)  # stddev can be changed for different problems
    b_init = tf.constant_initializer(value=0.0)  # value always zeros
    gamma_init = tf.random_normal_initializer(1, 0.02)  # use default value

    with tf.variable_scope("u_net",
                           reuse=reuse):  # define scope names of tensors, making thing convenient in monitoring tools, like tensorboard
        tl.layers.set_name_reuse(reuse)  # reuse name when try to validate during training
        inputs = tl.layers.InputLayer(x, name='inputs')  # input layers

        # First two convolution (Downsampling)
        conv1 = tl.layers.Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                 b_init=b_init,
                                 name='conv1_1')  # convolution layers:   kernel 3*3, output_channel 64; activation function: relu
        conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                 b_init=b_init,
                                 name='conv1_2')  # convolution layers:   kernel 3*3, output_channel 64; activation function: relu
        conv1 = BatchNormLayer(conv1, is_train=is_train, gamma_init=gamma_init,
                               name='bn1')  # BN: used reference set of gammaused to standardization the output of previous layers, help raise the convergence rate of loss value
        pool1 = tl.layers.MaxPool2d(conv1, (2, 2), padding='SAME',
                                    name='pool1')  # maxpool layer: kernel: 2*2 (downsample the image)

        # Second two convolution (Downsampling)
        conv2 = tl.layers.Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                 b_init=b_init,
                                 name='conv2_1')
        conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                 b_init=b_init,
                                 name='conv2_2')
        conv2 = BatchNormLayer(conv2, is_train=is_train, gamma_init=gamma_init,
                               name='bn2')
        pool2 = tl.layers.MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')

        # Third two convolution (Downsampling)
        conv3 = tl.layers.Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                 b_init=b_init,
                                 name='conv3_1')
        conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                 b_init=b_init,
                                 name='conv3_2')
        conv3 = BatchNormLayer(conv3, is_train=is_train, gamma_init=gamma_init,
                               name='bn3')
        pool3 = tl.layers.MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')

        # Fouth two convolution (Downsampling)
        conv4 = tl.layers.Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                 b_init=b_init,
                                 name='conv4_1')
        conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                 b_init=b_init,
                                 name='conv4_2')
        conv4 = BatchNormLayer(conv4, is_train=is_train, gamma_init=gamma_init,
                               name='bn4')
        pool4 = tl.layers.MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')

        # Fifth two convolution (Downsampling)
        conv5 = tl.layers.Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                 b_init=b_init,
                                 name='conv5_1')
        conv5 = tl.layers.Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                 b_init=b_init,
                                 name='conv5_2')
        conv5 = BatchNormLayer(conv5, is_train=is_train, gamma_init=gamma_init,
                               name='bn5')

        print(" * After conv: %s" % conv5.outputs)
        # First two convolution (upsampling)
        up4 = tl.layers.DeConv2d(conv5, 512, (3, 3),
                                 out_size=[tf.to_int32(tf.ceil(tf.shape(x)[1] / 8)),
                                           tf.to_int32(tf.ceil(tf.shape(x)[2] / 8))],
                                 strides=(2, 2),
                                 padding='SAME', act=tf.nn.relu, W_init=w_init, b_init=b_init, name='deconv4')
        up4 = BatchNormLayer(up4, is_train=is_train, gamma_init=gamma_init,
                             name='ucov_bn4_1')
        up4 = tl.layers.ConcatLayer([up4, conv4], concat_dim=3, name='concat4')
        conv4 = tl.layers.Conv2d(up4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv4_1')
        conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                 b_init=b_init,
                                 name='uconv4_2')
        conv4 = BatchNormLayer(conv4, is_train=is_train, gamma_init=gamma_init,
                               name='ucov_bn4_2')
        # Second two convolution (upsampling)


        up3 = tl.layers.DeConv2d(conv4, 256, (3, 3),
                                 out_size=[tf.to_int32(tf.ceil(tf.shape(x)[1] / 4)),
                                           tf.to_int32(tf.ceil(tf.shape(x)[2] / 4))],
                                 strides=(2, 2),
                                 padding='SAME', act=tf.nn.relu, W_init=w_init, b_init=b_init, name='deconv3')
        up3 = BatchNormLayer(up3, is_train=is_train, gamma_init=gamma_init,
                             name='ucov_bn3_1')
        up3 = tl.layers.ConcatLayer([up3, conv3], concat_dim=3, name='concat3')
        conv3 = tl.layers.Conv2d(up3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv3_1')
        conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                 b_init=b_init,
                                 name='uconv3_2')
        conv3 = BatchNormLayer(conv3, is_train=is_train, gamma_init=gamma_init,
                               name='ucov_bn3_2')

        up2 = tl.layers.DeConv2d(conv3, 128, (3, 3),
                                 out_size=[tf.to_int32(tf.ceil(tf.shape(x)[1] / 2)),
                                           tf.to_int32(tf.ceil(tf.shape(x)[2] / 2))],
                                 strides=(2, 2),
                                 padding='SAME', act=tf.nn.relu, W_init=w_init, b_init=b_init, name='deconv2')
        up2 = BatchNormLayer(up2, is_train=is_train, gamma_init=gamma_init,
                             name='ucov_bn2_1')
        up2 = tl.layers.ConcatLayer([up2, conv2], concat_dim=3, name='concat2')
        conv2 = tl.layers.Conv2d(up2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv2_1')
        conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                 b_init=b_init,
                                 name='uconv2_2')
        conv2 = BatchNormLayer(conv2, is_train=is_train, gamma_init=gamma_init,
                               name='ucov_bn2_2')

        up1 = tl.layers.DeConv2d(conv2, 64, (3, 3),
                                 out_size=[tf.to_int32(tf.shape(x)[1]), tf.to_int32(tf.shape(x)[2])],
                                 strides=(2, 2),
                                 padding='SAME', act=tf.nn.relu, W_init=w_init, b_init=b_init, name='deconv1')
        up1 = BatchNormLayer(up1, is_train=is_train, gamma_init=gamma_init,
                             name='ucov_bn1_1')
        up1 = tl.layers.ConcatLayer([up1, conv1], concat_dim=3, name='concat1')
        conv1 = tl.layers.Conv2d(up1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv1_1')
        conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                 b_init=b_init,
                                 name='uconv1_2')
        conv1 = BatchNormLayer(conv1, is_train=is_train, gamma_init=gamma_init,
                               name='ucov_bn1_2')

        conv1 = tl.layers.Conv2d(conv1, n_out, (1, 1), act=None, name='uconv1')

        out = tf.add(conv1.outputs, inputs.outputs, name='output')

        return out
