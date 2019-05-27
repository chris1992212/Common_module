import tensorflow as tf
import Data_utils
import net_utils
"""
This module is used to batch process all network 
"""
File_net = [1]
batch_valid = Data_utils.tfrecord_read_dataset(1, config.FE_size_ori, config.PE_size_ori, config.tfrecord_test,
                                               self.c_dim, False)
for k in range(File_net):

    restore_net_meta = 'Saved_model\\'+'Net'+str(k)+'\good\\train_model.ckpt.meta'
    restore_net_param = 'Saved_model\\'+ 'Net'+str(k)+'\good\\train_model.ckpt'
    tf.reset_default_graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(restore_net_meta)
        new_saver.restore(sess, restore_net_param)
        y = tf.get_collection('output')[0]
        graph = tf.get_default_graph()
        input = graph.get_operation_by_name('images').outputs[0]
        inferences = net_utils.create_model('u_net_model', low_res_holder, n_out=1, is_train=False, reuse=False)

        for ep in range(288):
            batch_xs_validation, batch_ys_validation = sess.run(batch_valid) # Get testing data for every iteration
            net_out = sess.run(y, feed_dict={input: batch_xs_validation
                                             })
