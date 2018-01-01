# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 14:09:27 2017

@author: jmanuel.navarro
"""
# Imports
import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

np.random.seed(42)

from keras.datasets import imdb

# DATASET LOADING AND PREPARATION

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                     num_words=None,
                                                     skip_top=0,
                                                     maxlen=None,
                                                     seed=113,
                                                     start_char=1,
                                                     oov_char=2,
                                                     index_from=3)

# x: each review is encoded as a sequence of indexes, corresponding to the 
# words in the review. The words are ordered by frequency, so the integer 1 
# corresponds to the most frequent word ("the")
# y: each review, comes with a label. A label of 0 is given to a negative review, 
#and a label of 1 is given to a positive review.

# Transform x and y into one hot encoding
 
# One-hot encoding the output into vector mode, each of length 1000
tokenizer = Tokenizer(num_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print(x_train[0])
# One-hot encoding the output
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape)
print(y_test.shape)

# BUILD KERAS MODEL
# Model definition
model = Sequential()
model.add(Dense(32, input_dim=x_train.shape[0]))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
# Mdodel compilation
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])

# MODEL TRAINING
model.fit(x_train, y_train, epochs=200, verbose=0)

# MODEL EVALUATION
model.evaluate('accuracy')


