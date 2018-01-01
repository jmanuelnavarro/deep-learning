# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 13:47:48 2017

@author: jmanuel.navarro
"""
import numpy as np
import pandas as pd
import keras

# PREPARE DATA

# Load data
data = pd.read_csv('.\student_data.csv')

# Transform rank values into dummy columns
rank_dummies = pd.get_dummies(data['rank'],prefix='rank')
one_hot_data = pd.concat([data,rank_dummies],axis=1)
one_hot_data = one_hot_data.drop(['rank'],axis=1)

# Scale gre and gpa
processed_data = one_hot_data
processed_data['gre'] = processed_data['gre']/800
processed_data['gpa'] = processed_data['gpa']/4

# Split dataa into training and testing(10%)
sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
train_data = processed_data.iloc[sample]
test_data = processed_data.drop(sample,axis=0)

# Split data into features (X) and targets (Y)
features = np.array(train_data.drop('admit', axis=1))
targets = np.array(keras.utils.to_categorical(train_data['admit'], 2))
features_test = np.array(test_data.drop('admit', axis=1))
targets_test = np.array(keras.utils.to_categorical(test_data['admit'], 2))

# DEFINE THE MODEL
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils

# Build the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(6,)))
model.add(Dropout(.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(2, activation='softmax'))

# Compiling the model
model.compile(loss = 'mean_squared_error', optimizer='SGD', metrics=['accuracy'])
model.summary()

# TRAIN THE MODEL
model.fit(features, targets, epochs=1000, batch_size=100, verbose=0)

# EVALUATE THE MODEL
score = model.evaluate(features, targets)
print("\n Training Accuracy:", score[1])
score = model.evaluate(features_test, targets_test)
print("\n Testing Accuracy:", score[1])





