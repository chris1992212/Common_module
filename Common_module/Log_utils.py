import tensorlayer as tl

""" This script is used to record the variable during training process"""

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
def log_record_txt(config,filename):
    log_dir = "log_{}".format(filename)
    tl.files.exists_or_mkdir(log_dir)
    log_all, log_all_filename = logging_setup(log_dir)
    log_config(log_all_filename, config)


def log_record_image(epoch, config,filename,list_test_pred_loss, list_test_src_loss, list_train_pred_loss, list_train_src_loss):


    self.log_all.debug(log)
    if epoch % 10000 == 0:
        saveMat.savemat(os.path.join(config.save_MODEL_file, 'list_test_pred_loss2.mat'),
                        mdict={'list_test_pred_loss2': list_test_pred_loss})
        saveMat.savemat(os.path.join(config.save_MODEL_file, 'list_test_src_loss2.mat'),
                        mdict={'list_test_src_loss2': list_test_src_loss})
        saveMat.savemat(os.path.join(config.save_MODEL_file, 'list_train_pred_loss2.mat'),
                        mdict={'list_train_pred_loss2': list_train_pred_loss})
        saveMat.savemat(os.path.join(config.save_MODEL_file, 'list_train_src_loss2.mat'),
                        mdict={'list_train_src_loss2': list_train_src_loss})
        plt.plot(range(len(list_test_pred_loss)), list_test_pred_loss, 'r',
                 range(len(list_test_pred_loss)), list_test_src_loss, 'y',
                 range(len(list_test_pred_loss)), list_train_pred_loss, 'b',
                 range(len(list_test_pred_loss)), list_train_src_loss, 'g')
        plt.savefig(os.path.join(config.save_MODEL_file, 'training.jpg'))