import tensorflow as tf
import Data_utils
import numpy as np
test_file = 'F:\matlab\Data_address_cml\BrainQuant_AI\Data\\train\\rawdata\Accel_factor4_2\\6channel\chenmingliang'
test_data = Data_utils.data_inputs(test_file,384,288,6)
# im1 = tf.placeholder(tf.float32,[None, 384, 288, 6],name = 'img1')
# im2 = tf.placeholder(tf.float32,[None, 384, 288, 6],name = 'img2')
im1 = tf.placeholder(tf.float32,[None, 384, 288, 1],name = 'img1')
im2 = tf.placeholder(tf.float32,[None, 384, 288, 1],name = 'img2')
ssim_value = tf.image.ssim(im1, im2, max_val=1.0)
ssim_values = np.zeros(6)
with tf.Session() as sess:
    imx, imy = next(test_data)

    for i in range(6):
        ssim_values[i] = sess.run(ssim_value, feed_dict= {im1: imx[:,:,:,i,np.newaxis], im2: imy[:,:,:,i,np.newaxis]})
        print('ssim value is %f'% ssim_values[i])
    # ssim_values[1] = sess.run(ssim_value,
    #                           feed_dict={im1: imx[:, :, :, :], im2: imy[:, :, :, :]})
    print('mean ssim value is %f' % ssim_values.mean())
