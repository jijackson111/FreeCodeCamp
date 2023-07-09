from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf
import matplotlib.pyplot as plt

# Get inputs
dftrain_inp = input('Copy and paste .csv link to training data: ')
dfeval_inp = input('Copy and paste .csv link to testing data: ')
dftrain = pd.read_csv(dftrain_inp)
dfeval = pd.read_csv(dfeval_inp)
col_del = input('Enter name of column in datasets to remove: ')
y_train = dftrain.pop(col_del)
y_eval = dfeval.pop(col_del)

# Categorical Columns
CATEGORICAL_COLUMNS = []
def c_cols():
    cc_lbl = input('Give list of categorical column names, separated by a comma and a space: ')
    cc_list = cc_lbl.split(', ')
    
    for elem in cc_list:
        CATEGORICAL_COLUMNS.append(elem)
    else:
        print('Is this all correct?', CATEGORICAL_COLUMNS)
        yn = input('\ny/n?: ')
        
        if yn == 'y':
            return None
        else:
            c_cols()
c_cols()

# Numeric columns
NUMERIC_COLUMNS = []
def n_cols():
    nc_lbl = input('Give list of numeric column names, separated by a comma and a space: ')
    nc_list = nc_lbl.split(', ')
    
    for elem in nc_list:
        NUMERIC_COLUMNS.append(elem)
    else:
        print('Is this all correct?', NUMERIC_COLUMNS)
        yn = input('\ny/n?: ')
       
        if yn == 'y':
            return None
        else:
            n_cols()
n_cols()
    
# Iterate over lists to put in next list
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocab = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))
for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
    
# Input function inputs
epc = int(input('Enter the number of desired epochs: '))
shuf = bool(input('Enter True to shuffle, False to not: '))
bsn = int(input('Enter batch size to use: '))

# Input function
def make_input_fn(data_df, label_df, num_epochs=epc, shuffle=shuf, batch_size=bsn):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Creating the model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# Training the model
linear_est.train(train_input_fn) 
result = linear_est.evaluate(eval_input_fn)  
clear_output() 
print(result['accuracy'])  

# Predict
pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
ploot = probs.plot(kind='hist', bins=20, title='predicted probabilities')
