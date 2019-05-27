"""This script is used to monitor the training, including the change of loss, other variables"""
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import numpy as np

def sav2csv(savFile, savdata):
    """
    This function is used to save the list to specific csv files, always curves of training data or validation data during training
    :return:
    """
    with open(savFile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(savdata)

def Loss_curves(path):
    """
    This function is used to read the .csv file to get the loss_curves.png
    :return:
    """
    with open(path, "r") as f:
        loss_curve = csv.reader(f)
        for loss in loss_curve:
            x_axis = np.arange(0, 1000*len(loss), 1000)
            plt.plot(x_axis,loss)
            label = ['Loss Curve']
            plt.legend(label, loc='upper right')
            plt.show()


def main(_):
    Loss_curves('Loss_curve.csv')
if __name__=='__main__':
    tf.app.run()