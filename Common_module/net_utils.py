import tensorflow.contrib.slim as slim
import tensorlayer as tl
from tensorlayer.layers import *
# from loss_func import  *


def create_model(name, patches, n_out, is_train=False, reuse=False):
    if name == 'srcnn':
        return srcnn_935(patches, reuse=reuse)
    elif name == 'vgg7':
        return vgg7(patches)
    elif name == 'vgg_deconv_7':
        return vgg_deconv_7(patches)
    elif name == 'u_net_model':
        return u_net_model(patches, n_out = n_out,is_train=is_train, reuse=reuse)
    elif name == 'srcnn_9751':
        return srcnn_9751(patches)
    elif name == 'SRresnet':
        return SRresnet(patches, n_out=n_out, is_train=is_train, reuse=reuse)
    elif name == 'EDSR':
        return EDSR(patches)
    elif name == 'SRDENSE':
        return SRDENSE(patches, n_out=n_out, is_train=is_train, reuse=reuse)

def u_net_model(images, n_out = 1, is_train = False, reuse = False):
        """

        :param images: input of model, always undersampled images
        :param is_train: to determine whether Batch Normalization
        :param reuse: reuse name of net
        :param n_out: dimension of output

        :return:
        """
        n_out = n_out
        x = images
        _, nx, ny, nz = x.get_shape().as_list()

        w_init = tf.truncated_normal_initializer(stddev=0.01)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1, 0.02)

        with tf.variable_scope("u_net", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            inputs = tl.layers.InputLayer(x, name='inputs')

            conv1 = tl.layers.Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv1_1')
            conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv1_2')
            conv1 = BatchNormLayer(conv1, is_train=is_train, gamma_init=gamma_init,
                                   name='bn1')
            pool1 = tl.layers.MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')


            conv2 = tl.layers.Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv2_1')
            conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv2_2')
            conv2 = BatchNormLayer(conv2, is_train=is_train, gamma_init=gamma_init,
                                   name='bn2')
            pool2 = tl.layers.MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')



            conv3 = tl.layers.Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv3_1')
            conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv3_2')
            conv3 = BatchNormLayer(conv3, is_train=is_train, gamma_init=gamma_init,
                                   name='bn3')
            pool3 = tl.layers.MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')


            conv4 = tl.layers.Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv4_1')
            conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv4_2')
            conv4 = BatchNormLayer(conv4, is_train=is_train, gamma_init=gamma_init,
                                   name='bn4')
            pool4 = tl.layers.MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')


            conv5 = tl.layers.Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv5_1')
            conv5 = tl.layers.Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv5_2')
            conv5 = BatchNormLayer(conv5, is_train=is_train, gamma_init=gamma_init,
                                   name='bn5')


            print(" * After conv: %s" % conv5.outputs)

            up4 = tl.layers.DeConv2d(conv5, 512, (3, 3),
                                     out_size=[tf.to_int32(tf.ceil(tf.shape(x)[1] / 8)), tf.to_int32(tf.ceil(tf.shape(x)[2] / 8))],
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


            up3 = tl.layers.DeConv2d(conv4, 256, (3, 3),
                                     out_size=[tf.to_int32(tf.ceil(tf.shape(x)[1] / 4)), tf.to_int32(tf.ceil(tf.shape(x)[2] / 4))],
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
                                     out_size=[tf.to_int32(tf.ceil(tf.shape(x)[1] / 2)), tf.to_int32(tf.ceil(tf.shape(x)[2] / 2))],
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
            # input = inputs.outputs
            ######## -------------------------Data fidelity--------------------------------##########
            # for contrast in range(n_out):
            #     k_conv3 = utils.Fourier(conv1[:,:,:,contrast], separate_complex=False)
            #     mask = np.ones((batch_size, nx, ny))
            #     mask[:,:, 1:ny:3] = 0
            #     mask = np.fft.ifftshift(mask)
            #     # convert to complex tf tensor
            #     DEFAULT_MAKS_TF = tf.cast(tf.constant(mask), tf.float32)
            #     DEFAULT_MAKS_TF_c = tf.cast(DEFAULT_MAKS_TF, tf.complex64)
            #     k_patches = utils.Fourier(input[:,:,:,contrast], separate_complex=False)
            #     k_space = k_conv3 * DEFAULT_MAKS_TF_c + k_patches*(1-DEFAULT_MAKS_TF_c)
            #     out = tf.ifft2d(k_space)
            #     out = tf.abs(out)
            #     out = tf.reshape(out, [batch_size, nx, ny, 1])
            #     if contrast == 0 :
            #         final_output = out
            #     else:
            #         final_output = tf.concat([final_output,out],3)
            ########-------------------------end------------------------------------###########3
            # print(" * Output: %s" % conv1.outputs)
            # outputs = tl.act.pixel_wise_softmax(conv1.outputs)
            return out

def u_net(x, n_out=12, reuse=False):  # Do I need to change n_out here ???
    _, nx, ny, nz = x.get_shape().as_list()
    # mean_x = tf.reduce_mean(x)
    # x = x - mean_x
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = tl.layers.InputLayer(x, name='inputs')
        # inputs = tf.layers.batch_normalization(inputs =inputs.outputs, axis= -1, training = True)
        # inputs = tl.layers.InputLayer(inputs, name ='input_batch_layer')

        conv1 = tl.layers.Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv1_1')
        conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv1_2')
        pool1 = tl.layers.MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')

        net1 = conv1.outputs
        # variable_summaries(net1, 'net_1')
        # tf.summary.image('net_1',net1[:,:,:,1],10)

        conv2 = tl.layers.Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv2_1')
        conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv2_2')
        pool2 = tl.layers.MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')

        net2 = conv2.outputs
        # variable_summaries(net2, 'net_2')
        # tf.summary.image('net_2',net2,10)

        conv3 = tl.layers.Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv3_1')
        conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv3_2')
        pool3 = tl.layers.MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')

        net3 = conv3.outputs
        # variable_summaries(net3, 'net_3')
        # tf.summary.image('net_3',net3,10)

        conv4 = tl.layers.Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv4_1')
        conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv4_2')
        pool4 = tl.layers.MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')

        net4 = conv4.outputs
        # variable_summaries(net4, 'net_4')
        # tf.summary.image('net_4',net4,10)

        conv5 = tl.layers.Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv5_1')
        conv5 = tl.layers.Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv5_2')

        net5 = conv5.outputs
        # variable_summaries(net5, 'net_5')
        # tf.summary.image('net_5',net5,10)

        print(" * After conv: %s" % conv5.outputs)

        up4 = tl.layers.DeConv2d(conv5, 512, (3, 3), out_size=(nx / 8, ny / 8), strides=(2, 2),
                                 padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv4')
        up4 = tl.layers.ConcatLayer([up4, conv4], concat_dim=3, name='concat4')
        conv4 = tl.layers.Conv2d(up4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv4_1')
        conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv4_2')

        netup4 = conv4.outputs
        # variable_summaries(netup4, 'netup_4')
        # tf.summary.image('netup_4',netup4,10)

        up3 = tl.layers.DeConv2d(conv4, 256, (3, 3), out_size=(nx / 4, ny / 4), strides=(2, 2),
                                 padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv3')
        up3 = tl.layers.ConcatLayer([up3, conv3], concat_dim=3, name='concat3')
        conv3 = tl.layers.Conv2d(up3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv3_1')
        conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv3_2')

        netup3 = conv3.outputs
        # variable_summaries(netup3, 'netup_3')
        # tf.summary.image('netup_3',netup3,10)

        up2 = tl.layers.DeConv2d(conv3, 128, (3, 3), out_size=(nx / 2, ny / 2), strides=(2, 2),
                                 padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv2')
        up2 = tl.layers.ConcatLayer([up2, conv2], concat_dim=3, name='concat2')
        conv2 = tl.layers.Conv2d(up2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv2_1')
        conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv2_2')

        netup2 = conv2.outputs
        # variable_summaries(netup2, 'netup_2')
        # tf.summary.image('netup_2',netup2,10)

        up1 = tl.layers.DeConv2d(conv2, 64, (3, 3), out_size=(nx / 1, ny / 1), strides=(2, 2),
                                 padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv1')
        up1 = tl.layers.ConcatLayer([up1, conv1], concat_dim=3, name='concat1')
        conv1 = tl.layers.Conv2d(up1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv1_1')
        conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv1_2')

        netup1 = conv1.outputs
        # variable_summaries(netup1, 'netup_1')
        # tf.summary.image('netup_1',netup1,10)

        conv1 = tl.layers.Conv2d(conv1, n_out, (1, 1), act=None, name='uconv1')
        # tf.summary.image('final', tf.cast(conv1.outputs,tf.uint8), 1)
        # tf.summary.image('final', conv1.outputs, 10)
        conv1 = tf.add(conv1.outputs, inputs.outputs)

        # print(" * Output: %s" % conv1.outputs)
        # outputs = tl.act.pixel_wise_softmax(conv1.outputs)
        return conv1

def srcnn_9751(patches, name='zqq'):
    with tf.variable_scope(name):
        # upscaled_patches = tf.image.resize_bicubic(patches, [INPUT_SIZE, INPUT_SIZE], True)
        conv1 = conv2d(patches, 9, 9, 64, padding='SAME', name='conv1')
        relu1 = relu(conv1, name='relu1')
        dd = tf.transpose(relu1, perm=[3, 1, 2, 0])
        # tf.summary.image('conv1', dd, 10)

        conv2 = conv2d(relu1, 7, 7, 32, padding='SAME', name='conv2')
        relu2 = relu(conv2, name='relu2')
        dd = tf.transpose(relu2, perm=[3, 1, 2, 0])
        # tf.summary.image('conv2', dd, 10)
        conv3 = conv2d(relu2, 1, 1, 16, padding='SAME', name='conv2')
        relu3 = relu(conv3, name='relu3')

        return conv2d(relu3, 5, 5, NUM_CHENNELS, padding='SAME', name='conv3')

def srcnn_935(patches, name='srcnn', reuse=False):
    batch_size, nx, ny, nz = patches.get_shape().as_list()
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = tl.layers.InputLayer(patches, name='inputs')
        conv1 = tl.layers.Conv2d(inputs, 64, (9, 9), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv1')
        conv2 = tl.layers.Conv2d(conv1, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv2')
        conv3 = tl.layers.Conv2d(conv2, 1, (5, 5), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='conv3')
        # upscaled_patches = tf.image.resize_bicubic(patches, [INPUT_SIZE, INPUT_SIZE], True)
        ######## Data fidelity
        # k_conv3 = utils.Fourier(conv3.outputs, separate_complex=False)
        # mask = np.ones((batch_size, nx, ny))
        # mask[:, np.array(nx / 4 + 1, dtype='int8'):np.array(nx * 3 / 4, dtype='int8'), :] = 0
        # mask = np.fft.ifftshift(mask)
        # # convert to complex tf tensor
        # DEFAULT_MAKS_TF = tf.cast(tf.constant(mask), tf.float32)
        # DEFAULT_MAKS_TF_c = tf.cast(DEFAULT_MAKS_TF, tf.complex64)
        # k_patches = utils.Fourier(patches, separate_complex=False)
        # k_space = k_conv3 * DEFAULT_MAKS_TF_c + k_patches
        # out = tf.ifft2d(k_space)
        # out = tf.abs(out)
        # out = tf.reshape(out, [batch_size, nx, ny, 1])
        # k_conv3[:,nx/4+1:nx*3/4,ny/4+1:ny*3/4,:] = k_patches[:,nx/4+1:nx*3/4,ny/4+1:ny*3/4,:]

        return conv3.outputs

def SRresnet(t_image, n_out=12, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        Input = InputLayer(t_image, name='in')
        n = Conv2d(Input, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n

        # B residual blocks
        for i in range(32):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, 'b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, 'add3')
        # B residual blacks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n256s1/1')
        # n = SubpixelConv2d(n, scale=1, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n256s1/2')
        # n = SubpixelConv2d(n, scale=1, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

        # n = Conv2d(n, n_out, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        n = Conv2d(n, n_out, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        n = tf.add(n.outputs, Input.outputs)

        return n

def SRresnet_no_batch(t_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n

        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c1/%s' % i)
            # nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c2/%s' % i)
            # nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, 'b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, 'add3')
        # B residual blacks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n256s1/1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')

        # n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n256s1/2')
        # n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

        n = Conv2d(n, NUM_CHENNELS, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, name='out')
        tf.summary.image('final', tf.cast(n.outputs, tf.uint8), 1)
        return n.outputs

def vgg7(patches, name='vgg7'):
    """
    模型的输出
    :param patches: input patches to improve resolution. must has format of
        [batch_size, patch_height, patch_width, patch_chennels]
    :param name: the name of the network
    :return: the RSCNN inference function
    """
    with tf.variable_scope(name):
        # upscaled_patches = tf.image.resize_bicubic(patches, [INPUT_SIZE, INPUT_SIZE], True)
        conv1 = conv2d(patches, 3, 3, 32, padding='SAME', name='conv1')
        lrelu1 = leaky_relu(conv1, name='leaky_relu1')
        conv2 = conv2d(lrelu1, 3, 3, 32, padding='SAME', name='conv2')
        lrelu2 = leaky_relu(conv2, name='leaky_relu2')
        conv3 = conv2d(lrelu2, 3, 3, 64, padding='SAME', name='conv3')
        lrelu3 = leaky_relu(conv3, name='leaky_relu3')
        conv4 = conv2d(lrelu3, 3, 3, 64, padding='SAME', name='conv4')
        lrelu4 = leaky_relu(conv4, name='leaky_relu4')
        conv5 = conv2d(lrelu4, 3, 3, 128, padding='SAME', name='conv5')
        lrelu5 = leaky_relu(conv5, name='leaky_relu5')
        conv6 = conv2d(lrelu5, 3, 3, 128, padding='SAME', name='conv6')
        lrelu6 = leaky_relu(conv6, name='leaky_relu6')
        return conv2d(lrelu6, 3, 3, NUM_CHENNELS, padding='SAME', name='conv_out')

def vgg_deconv_7(patches, name='vgg_deconv_7'):
    with tf.variable_scope(name):
        conv1 = conv2d(patches, 3, 3, 16, padding='SAME', name='conv1')
        lrelu1 = leaky_relu(conv1, name='leaky_relu1')
        conv2 = conv2d(lrelu1, 3, 3, 32, padding='SAME', name='conv2')
        lrelu2 = leaky_relu(conv2, name='leaky_relu2')
        conv3 = conv2d(lrelu2, 3, 3, 64, padding='SAME', name='conv3')
        lrelu3 = leaky_relu(conv3, name='leaky_relu3')
        conv4 = conv2d(lrelu3, 3, 3, 128, padding='SAME', name='conv4')
        lrelu4 = leaky_relu(conv4, name='leaky_relu4')
        conv5 = conv2d(lrelu4, 3, 3, 128, padding='SAME', name='conv5')
        lrelu5 = leaky_relu(conv5, name='leaky_relu5')
        conv6 = conv2d(lrelu5, 3, 3, 256, padding='SAME', name='conv6')
        lrelu6 = leaky_relu(conv6, name='leaky_relu6')

        batch_size = int(lrelu6.get_shape()[0])
        rows = int(lrelu6.get_shape()[1])
        cols = int(lrelu6.get_shape()[2])
        channels = int(patches.get_shape()[3])
        # to avoid chessboard artifacts, the filter size must be dividable by the stride
        return deconv2d(lrelu6, 4, 4, [batch_size, rows, cols, channels], stride=(1, 1), name='deconv_out')

def EDSR(patches, feature_size=64, num_layers=16):
    print("Building EDSR...")
    # mean_x = tf.reduce_mean(patches)
    # image_input = patches - mean_x
    # mean_y = tf.reduce_mean(high_patches)
    # image_target = y - mean_y

    x = slim.conv2d(patches, feature_size, [3, 3])
    conv_1 = x

    """
            This creates `num_layers` number of resBlocks
            a resBlock is defined in the paper as
            (excuse the ugly ASCII graph)
            x
            |\
            | \
            |  conv2d
            |  relu
            |  conv2d
            | /
            |/
            + (addition here)
            |
            result
            """

    """
    Doing scaling here as mentioned in the paper:

    `we found that increasing the number of feature
    maps above a certain level would make the training procedure
    numerically unstable. A similar phenomenon was
    reported by Szegedy et al. We resolve this issue by
    adopting the residual scaling with factor 0.1. In each
    residual block, constant scaling layers are placed after the
    last convolution layers. These modules stabilize the training
    procedure greatly when using a large number of filters.
    In the test phase, this layer can be integrated into the previous
    convolution layer for the computational efficiency.'

    """
    scaling_factor = 1

    # Add the residual blocks to the model
    for i in range(num_layers):
        x = resBlock(x, feature_size, scale=scaling_factor)

    # One more convolution, and then we add the output of our first conv layer
    x = slim.conv2d(x, feature_size, [3, 3])
    x += conv_1

    # Upsample output of the convolution
    x = upsample(x, NUM_CHENNELS, Scale, feature_size, None)
    output = x
    mean_x = tf.reduce_mean(patches)
    f = tf.sqrt(tf.reduce_sum(tf.square(output + mean_x), axis=-1))
    ff = tf.reshape(f, [-1, 384, 384, 1])
    tf.summary.image("output_image", tf.cast(ff, tf.uint8))
    # f = tf.sqrt(tf.reduce_sum(tf.square(output[:,:,:,0:2] + mean_x), axis=-1))
    # ff = tf.reshape(f, [-1, 384, 384, 1])
    # tf.summary.image("output_image_echo1_FA1", tf.cast(ff, tf.uint8))
    # f = tf.sqrt(tf.reduce_sum(tf.square(output[:,:,:,2:4] + mean_x), axis=-1))
    # ff = tf.reshape(f, [-1, 384, 384, 1])
    # tf.summary.image("output_image_echo2_FA1", tf.cast(ff, tf.uint8))
    # f = tf.sqrt(tf.reduce_sum(tf.square(output[:,:,:,4:6] + mean_x), axis=-1))
    # ff = tf.reshape(f, [-1, 384, 384, 1])
    # tf.summary.image("output_image_echo3_FA1", tf.cast(ff, tf.uint8))
    # f = tf.sqrt(tf.reduce_sum(tf.square(output[:,:,:,6:8] + mean_x), axis=-1))
    # ff = tf.reshape(f, [-1, 384, 384, 1])
    # tf.summary.image("output_image_echo1_FA2", tf.cast(ff, tf.uint8))
    # f = tf.sqrt(tf.reduce_sum(tf.square(output[:,:,:,8:10] + mean_x), axis=-1))
    # ff = tf.reshape(f, [-1, 384, 384, 1])
    # tf.summary.image("output_image_echo2_FA2", tf.cast(ff, tf.uint8))
    # f = tf.sqrt(tf.reduce_sum(tf.square(output[:,:,:,10:12] + mean_x), axis=-1))
    # ff = tf.reshape(f, [-1, 384, 384, 1])
    # tf.summary.image("output_image_echo3_FA2", tf.cast(ff, tf.uint8))
    return output









