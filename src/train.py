from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# function will take container data as input and return LSTM ready input format data. 
# The window length and overlap can be set and a timeseries generator is used.
def create_windows_smoothed(dfs, length, batch_size, stride):
    features = []
    targets = []
    X = []
    y = []

    # Height as feature and target
    for df in dfs:
        height = df['inter_pol'].to_numpy().tolist()
        
        # apply TimeSeriesGenerator
        ts_generator = TimeseriesGenerator(height,height,length=length, batch_size=batch_size, stride=stride)

        for j in range(len(ts_generator)):
            features.append(ts_generator[j][0])
            targets.append(ts_generator[j][1])
    
    #reshape data for neural network
    for i in range(len(features)):
        x = np.reshape(features[i], (length,1))
        X.append(x)
    X = np.array(X)
    
    y = np.array(targets)
    return X, y


# function creates train/validation/test split on data
def create_train_val_test_split(X,y):
    #Split data into train & test set & validation set 
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X_tr, y_tr, test_size=0.2)
    
    return X_train, y_train, X_test, y_test, X_val, y_val