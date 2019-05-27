import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import os
from tensorflow.python.framework import graph_util


""" Modle_filename: Must be use 反斜杆，so that path can be found"""

Model_filename_meta =os.path.join('G:\AI\\Net_model\class1\\Net1\\ph_net','good','train_model.ckpt.meta')

# Model_filename_meta = "D://SR_crop//model_Amp_6channel_bn_7_12//u_net_bn_new_2//good//mymodel.meta"
Model_filename = os.path.join('G:\AI\\Net_model\\class1\\Net1\\ph_net\\good\\train_model.ckpt')
# Model_filename = os.path.join('Good_model_for_Amp','model_Amp_6channel_bn_7_12','u_net_bn_new_2','good','mymodel')
def save_model_pb(model_filename_meta,restore_net_param):

    with tf.Session() as sess:

        saver = tf.train.import_meta_graph(model_filename_meta)
        saver.restore(sess, restore_net_param)
        graph = tf.get_default_graph()
        validation_images = graph.get_operation_by_name('validation_images').outputs[0]
        y_pred = graph.get_operation_by_name('u_net_1/output').outputs[0]



        # 保存图
        tf.train.write_graph(sess.graph_def, './pb_dir_output/pb_model', 'mymodel_final.pbtxt')
        # 把图和参数结构一起
        freeze_graph.freeze_graph('pb_dir_output/pb_model/mymodel_final.pbtxt',
                                  '',
                                  False,
                                  Model_filename,
                                  'u_net/output',
                                  'save/restore_all',
                                  'save/Const:0',
                                  'pb_dir_output/pb_model/model_frozen.pb',
                                  False,
                                  "")
    print("done")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ' 0'
    save_model_pb(Model_filename_meta, Model_filename)