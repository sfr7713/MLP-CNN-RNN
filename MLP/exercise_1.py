"""
Numpy provides support for large, multi-dimensional arrays and matrices, along with a large collection 
of high-level mathematical functions to operate on these arrays.

Pandas offers data structures and operations for manipulating numerical tables and time series
"""
import numpy as np
import pandas as pd
import math 

"""
The core data structure of Keras is a model, a way to organize layers. 
The simplest type of model is the Sequential model, a linear stack of layers. 
"""
from keras.models import Sequential
from keras.optimizers import SGD

from sklearn import preprocessing
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Input, BatchNormalization
import tensorflow as tf

def helloworld():
    print("Hello, world!")
    # print("uncomment me")
    return


def clean_data_and_labels(df):
    ### YOUR CODE HERE ###
    # remove the samples (rows) with a NaN that occur anywhere in its features
    # Then remove the samples in which the diabetes label is 3
    # 1-2 lines
    
    # Turn race into dummy variables 
    #race_cols = ['Race_'+ str(i) for i in [1,2,3,4,6,7]]
    #df[race_cols] = pd.get_dummies(df.Race)
    #df = df.drop("Race", axis=1)
    
    #df['Race'] = df['Race'].astype('category') 
    df.dropna(inplace=True)
    df = df[df.Diabetes < 3] 

    ######################
    return df


def split_x_y(df, col = 'Diabetes'):
    """
    Problem with label encoding is that it assumes higher the categorical value, better the category.
    This is why we use one hot encoder to perform “binarization” of the category 
    and include it as a feature to train the model.
    """
    x = df.drop(col, axis=1)
    y = pd.get_dummies(df[col])
    return x, y


def preprocess_dataset(df):
    """
    Perform any preprocessing on your dataset here (i.e. data balancing,
    normalization, etc)
    You should return a dataframe.
    """
    ### YOUR CODE HERE ###
	
    # standardize features
    cols = df.columns.tolist()
    cols.remove('Race')
    if 'Diabetes' in cols:
	    cols.remove('Diabetes')
    scaler = preprocessing.StandardScaler().fit(df[cols])
    df[cols] = scaler.transform(df[cols])
	
	# turn categorical feature to dummies
    df = pd.get_dummies(df, prefix = 'Race', columns = ['Race'])
      
    ######################
    return df


def as_keras_metric(method):
    import functools
    from keras import backend as K
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper             
    


def build_model(input_dim):
    model = Sequential()
    ### YOUR CODE HERE ###
    # Build a your fully connected hidden layers
    # 1+ lines
    model.add(Dense(input_dim= input_dim, units= 32, activation="relu")) 
    model.add(BatchNormalization(momentum=0.99, center=True, scale=True))
    model.add(Dropout(0.5))
    
    model.add(Dense(20, activation="relu")) 
    model.add(BatchNormalization(momentum=0.99, center=True, scale=True))
    model.add(Dropout(0.5))
    
    model.add(Dense(10, activation="relu")) 
    model.add(BatchNormalization(momentum=0.99, center=True, scale=True))
    model.add(Dropout(0.5))
    # Build your final "readout" layer
    # 1 line
    model.add(Dense(units= 2, activation="softmax"))    

    ######################
    # The loss function and optimizer are provided for you.
    sgd = SGD(lr=0.001, decay=1e-7, momentum=0.9)  #Stochastic gradient descent
    auc_roc = as_keras_metric(tf.metrics.auc)
    model.compile(loss="categorical_crossentropy", optimizer= sgd , metrics=['accuracy', auc_roc])
    return model
