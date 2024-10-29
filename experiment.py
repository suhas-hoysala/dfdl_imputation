import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interp
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from DeepDRIM.DeepDRIM import direct_model1_squarematrix

def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-num_batches', type=int, required=True, default=None, help="Number of TF or the number of x file.")
    parser.add_argument('-data_path', required=True, default=None, help="The path that includes x file, y file and z file.")
    parser.add_argument('-output_dir', required=True, default="./output/", help="Indicate the path for output.")
    parser.add_argument('-cross_validation_fold_divide_file', default=None, help="A file that indicate how to divide the x file into three-fold. The file include three line, each line list the ID of the x files for the folder (split by ',')")
    parser.add_argument('-to_predict', default=False, help="True or False. Default is False, then the code will do cross-validation evaluation. If set to True, we need to indicate weight_path for a trained model and the code will do prediction based on the trained model.")
    parser.add_argument('-weight_path', default=None, help="The path for a trained model.")

    return parser.parse_args()

from consolidated_runs import run_scenic

class DirectModel(direct_model1_squarematrix):
    def __init__(self, num_batches=5, output_dir=None, data_path=None, predict_output_dir=None, method=None, method_name=None):
        # Initial setup
        self.data_augmentation = False
        self.batch_size = 32  
        self.epochs = 200  
        self.method_name = method_name
        self.output_dir = output_dir
        self.num_batches = num_batches
        self.data_path = data_path
        self.method = method
        self.num_classes = 2
        self.whole_data_TF = [i for i in range(self.num_batches)]
        self.x_train, self.y_train, self.z_train = None, None, None
        self.x_test, self.y_test, self.z_test = None, None, None

    def split_data(x_path, y_path, train_ratio=0.7, val_ratio=0.15):
        y_data = np.transpose(np.load(y_path, allow_pickle=True))
        x_data = np.transpose(np.load(x_path, allow_pickle=True))
        # Split data indices
        total_samples = x_data.shape[0]
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)

        x_train, y_train = x_data[:train_end], y_data[:train_end]
        x_val, y_val = x_data[train_end:val_end], y_data[train_end:val_end]
        x_test, y_test = x_data[val_end:], y_data[val_end:]

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def prepare_data_with_method(self, x_data, y_data, ind):
        print(f"Processing DS{str(ind)} set with {self.method_name}...")
        return self.method(x_data, y_data, ind)


    def run_scenic(self, x_data, y_data, z_data):
        # Apply SCENIC to data (train, test, valid)
        print("Running SCENIC on data split...")
        # Placeholder: Replace this with    atual SCENIC processing code
        # Assuming it modifies or enhances x_data, y_data, z_data in some way
        # For example:
        # x_data, y_data, z_data = scenic_pipeline(x_data, y_data, z_data)
        return x_data, y_data, z_data

    def load_data_TF2(self, i):
        individual_results = {}

        imp_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),  'SERGIO/imputation_data'))
        imp_data_45_fnames = [os.path.join(imp_data_dir, fname) for fname in os.listdir(imp_data_dir) if fname.startswith('DS' + str(i)) and '45' in fname]
        imp_data_clean_fnames = [os.path.join(imp_data_dir, fname) for fname in os.listdir(imp_data_dir) if fname.startswith('DS' + str(i)) and 'clean' in fname]

        imp_data_45 = imp_data_45_fnames[0]
        imp_data_clean = imp_data_clean_fnames[0]

        (x_train, y_train), (x_val, y_val), (x_test, y_test) = self.split_data(imp_data_clean, imp_data_45)

        self.x_train, self.y_train = self.prepare_data_with_method(x_train, y_train, i)
        self.x_val, self.y_val = self.prepare_data_with_method(x_val, y_val, i)
        self.x_test, self.y_test = self.prepare_data_with_method(x_test, y_test, i)

    def update_test_train_data(self, i):
        self.load_data_TF2(i)

    def train_and_test_model_dividePart_assignTForder(self,num_of_pair_ratio=1):
        datasets = range(1,4)
        for i in tqdm(datasets):
            print(f"--> Updating train and test data with {self.method_name} preprocessing for DS{i}...")
            self.update_test_train_data(i)
            self.construct_model(self.x_train)
            history = self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs,
                                        validation_split=0.2, shuffle=True, callbacks=self.callbacks_list)
            self.test_model(self.model, self.x_test, self.y_test, self.z_test, self.save_dir, history, None)


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
    args = get_args()
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
