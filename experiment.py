import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

parser = argparse.ArgumentParser(description="")

parser.add_argument('-num_batches', type=int, required=True, default=None, help="Number of TF or the number of x file.")
parser.add_argument('-data_path', required=True, default=None, help="The path that includes x file, y file and z file.")
parser.add_argument('-output_dir', required=True, default="./output/", help="Indicate the path for output.")
parser.add_argument('-cross_validation_fold_divide_file', default=None, help="A file that indicate how to divide the x file into three-fold. The file include three line, each line list the ID of the x files for the folder (split by ',')")
parser.add_argument('-to_predict', default=False, help="True or False. Default is False, then the code will do cross-validation evaluation. If set to True, we need to indicate weight_path for a trained model and the code will do prediction based on the trained model.")
parser.add_argument('-weight_path', default=None, help="The path for a trained model.")

args = parser.parse_args()

class DirectModel:
    def __init__(self, num_batches=5, output_dir=None, data_path=None, predict_output_dir=None):
        # Initial setup
        self.data_augmentation = False
        self.batch_size = 32  
        self.epochs = 200  
        self.model_name = 'my_model'
        self.output_dir = output_dir
        self.num_batches = num_batches
        self.data_path = data_path
        self.num_classes = 2
        self.whole_data_TF = [i for i in range(self.num_batches)]
        self.x_train, self.y_train, self.z_train = None, None, None
        self.x_test, self.y_test, self.z_test = None, None, None

    def run_scenic(self, x_data, y_data, z_data):
        # Apply SCENIC to data (train, test, valid)
        print("Running SCENIC on data split...")
        # Placeholder: Replace this with actual SCENIC processing code
        # Assuming it modifies or enhances x_data, y_data, z_data in some way
        # For example:
        # x_data, y_data, z_data = scenic_pipeline(x_data, y_data, z_data)
        return x_data, y_data, z_data

    def load_data_TF2(self, indel_list, data_path, num_of_pair_ratio=1, train=True): 
        xxdata_list, yydata, zzdata, count_set = [], [], [], [0]
        for i in indel_list:
            try:
                xdata = np.load(data_path + str(i) + '_xdata.npy')
                ydata = np.load(data_path + str(i) + '_ydata.npy')
                zdata = np.load(data_path + str(i) + '_zdata.npy')
                xxdata_list.extend(xdata), yydata.extend(ydata), zzdata.extend(zdata)
                count_set.append(count_set[-1] + len(ydata))
            except Exception as e:
                print(f"Error loading data for index {i}: {e}")
                continue

        # Run SCENIC processing on loaded data
        xxdata_list, yydata, zzdata = self.run_scenic(xxdata_list, yydata, zzdata)

        if train:
            self.x_train, self.y_train, self.z_train = np.array(xxdata_list), np.array(yydata), np.array(zzdata)
        else:
            self.x_test, self.y_test, self.z_test = np.array(xxdata_list), np.array(yydata), np.array(zzdata)

    def update_test_train_data(self, test_indel, epochs, num_of_pair_ratio=1):
        train_TF = [i for i in self.whole_data_TF if i not in test_indel]
        self.load_data_TF2(train_TF, self.data_path, num_of_pair_ratio, train=True)
        self.load_data_TF2(test_indel, self.data_path, num_of_pair_ratio, train=False)

    def train_and_test_model_dividePart_assignTForder(self, indel_list0, indel_list1, indel_list2, num_of_pair_ratio=1):
        for i, test_indel in enumerate([indel_list0, indel_list1, indel_list2]):
            print("--> Updating train and test data with SCENIC preprocessing...")
            self.update_test_train_data(test_indel, self.epochs, num_of_pair_ratio)
            self.construct_model(self.x_train)
            history = self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs,
                                     validation_split=0.2, shuffle=True, callbacks=self.callbacks_list)
            self.test_model(self.model, self.x_test, self.y_test, self.z_test, self.save_dir, history, test_indel)


def load_indel_lists_from_file(cross_validation_fold_divide_file):
    with open(cross_validation_fold_divide_file) as f:
        cross_fold = [line.strip().split(',') for line in f]
    indel_list0, indel_list1, indel_list2 = cross_fold
    print('indel_list0', indel_list0)
    print('indel_list1', indel_list1)
    print('indel_list2', indel_list2)
    return indel_list0, indel_list1, indel_list2

def main():
    tcs = DirectModel(num_batches=args.num_batches, data_path=args.data_path, output_dir=args.output_dir, method=args.method)
    indel_list0, indel_list1, indel_list2 = load_indel_lists_from_file(args.cross_validation_fold_divide_file)
    tcs.train_and_test_model_dividePart_assignTForder(indel_list0, indel_list1, indel_list2)

def main_predict():
    tcs = DirectModel(num_batches=args.num_batches, data_path=args.data_path, predict_output_dir=args.output_dir, method=args.method)
    tcs.predict_use_model(args.weight_path)

if __name__ == '__main__':
    if args.to_predict:
        if args.weight_path:
            main_predict()
        else:
            print("Require input trained model weight_path.")
    else:
        if args.cross_validation_fold_divide_file:
            main()
        else:
            print("Require input cross_validation_fold_divide_file")
