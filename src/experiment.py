# experiment.py
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
# import torch
# import torchvision.models as models
from models.resnet import Classifier_RESNET
import numpy as np
from data import read_dataset, normalize_data
import os
from sklearn.preprocessing import OneHotEncoder

class Experiment:
    def __init__(self, model):
        # self.__conf = config
        self.__name = model
    def run(self):
        model_name = self.__name
        output_directory = '../results/' + model_name + '/' 

        test_dir_df_metrics = output_directory + 'df_metrics.csv'

        print('Method: ', model_name)

        if os.path.exists(test_dir_df_metrics):
            print('Already done')
        # datasets_dict = read_dataset()
       
        datasets_dict = []
    
        file_name = '../data/' 
        train_ratio = 0.75
        validation_ratio = 0.15
        test_ratio = 0.10
        train_positive_ex1 = np.fromfile(file_name + 'exp1-train-400pos.bin', dtype=np.float32).reshape([400,5,768])
        train_negative_ex1 = np.fromfile(file_name + 'exp1-train-2000neg.bin', dtype=np.float32).reshape([2000,5,768])
        train_positive_ex2 = np.fromfile(file_name + 'exp2-train-400pos.bin', dtype=np.float32).reshape([400,5,768])
        train_negative_ex2 = np.fromfile(file_name + 'exp2-train-2000neg.bin', dtype=np.float32).reshape([2000,5,768])
        # Concatenate data for both experiments and classes
        X = np.concatenate([train_positive_ex1, train_negative_ex1, train_positive_ex2, train_negative_ex2], axis=0)
        # Create target labels: positive (1) and negative (0)
        y_positive_ex1 = np.ones(400)
        y_negative_ex1 = np.zeros(2000)
        y_positive_ex2 = np.ones(400)
        y_negative_ex2 = np.zeros(2000)

        y = np.concatenate([y_positive_ex1, y_negative_ex1, y_positive_ex2, y_negative_ex2])


        datasets_dict.extend([X.copy(), y.copy()])
        
        x_data = datasets_dict[0]
        y_data = datasets_dict[1]
    
        # train is now 75% of the entire data set
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=1 - train_ratio, stratify=y_data)
        
        # test is now 10% of the initial data set
        # validation is now 15% of the initial data set
        # x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))
        x_train_normalized,x_test_normalized = normalize_data(x_train, x_test)
        
        # transform the labels from integers to one hot vectors
        enc = OneHotEncoder(categories='auto')
        enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
        y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
        y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

        nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

        # # transform the labels from integers to one hot vectors
        # enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
        # enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
        # y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
        # y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

        # save orignal y because later we will use binary
        y_true = np.argmax(y_test, axis=1)


        input_shape = x_train.shape[1:]
        classifier = Classifier_RESNET(output_directory, input_shape, nb_classes, verbose=True)

        classifier.fit(x_train_normalized, y_train, x_test_normalized, y_test, y_true)
        
        