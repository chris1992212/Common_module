import tensorflow as tf


def L2_loss(pred, labels, name='L2_loss'):
    with tf.name_scope(name):
        l2_loss = tf.reduce_mean(tf.square(labels - pred))
        return l2_loss

def L1_loss(pred, labels, name = 'L1_loss'):
    with tf.name_scope(name):
        l1_loss = tf.reduce_mean(tf.abs(labels - pred))
        return l1_loss


def loss_SSIM(y_true, y_pred):
    ssim = tf.image.ssim(y_true,y_pred, max_val=1 )
    return tf.reduce_mean((1.0 - ssim)/2,name = 'ssim_loss')

def loss_MS_SSIM(y_true, y_pred):
    ms_ssim = tf.image.ssim_multiscale(y_true, y_pred, max_val= 1)
    return  tf.reduce_mean((1.0 - ms_ssim)/2,name = 'ssim_ms_loss')
def loss_SSIM_MAE(y_true, y_pred, name = 'SSIM_MAE_loss'):
    ssim = loss_SSIM(y_true,y_pred)
    ssim_mae_final = ssim + 5*L1_loss(y_true, y_pred)
    return ssim_mae_final