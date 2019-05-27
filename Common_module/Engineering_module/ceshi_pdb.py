# -*- coding:utf-8 -*-
import argparse
import tensorflow as tf
import Data_utils
from scipy.misc import imsave, imread, imresize
import os
import numpy as np
import scipy.io as sio

def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name= 'perfix',
            op_dict=None,
            producer_op_list=None
        )
    return graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="pb_dir_output/pb_model/model_frozen.pb", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()
    # 加载已经将参数固化后的图
    graph = load_graph(args.frozen_model_filename)

    # We can list operations
    # op.values() gives you a list of tensors it produces
    # op.name gives you the name
    # 输入,输出结点也是operation,所以,我们可以得到operation的名字
    for op in graph.get_operations():
        print(op.name, op.values())
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
    # 操作有:prefix/Placeholder/inputs_placeholder
    # 操作有:prefix/Accuracy/predictions
    # 为了预测,我们需要找到我们需要feed的tensor,那么就需要该tensor的名字
    # 注意prefix/Placeholder/inputs_placeholder仅仅是操作的名字,prefix/Placeholder/inputs_placeholder:0才是tensor的名字
    x = graph.get_tensor_by_name('perfix/low:0')
    y = graph.get_tensor_by_name('perfix/u_net/output:0')
    # high_res_holder = graph.get_tensor_by_name('perfix/low:0')
    testing_filename = os.path.join('Data','train', 'ceshi', 'one_channel')
    # testing_filename = os.path.join('Data', 'test', 'xujun', 'PE_324')
    test_epoch = Data_utils.data_inputs(testing_filename, 384,324, 1, False, 1)
    # testing_loss = SRCNN_models.loss(y, high_res_holder, name='testing_loss', weights_decay=0)
    # mse = np.zeros((288,1))
    out = np.zeros((384,324,288,1))

    with tf.Session(graph=graph) as sess:
        for i in range(288):

            batch_xs, batch_ys = next(test_epoch)

            out[:, :, i, :] = sess.run(y, feed_dict={
                x:batch_xs
            })
        sio.savemat('recon.mat', {'recon': out})
    # print("%d", mse.mean())
