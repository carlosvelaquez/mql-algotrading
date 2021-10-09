from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix

import os
import datetime

import tensorflow as tf
import numpy as np
import pandas as pd

VAL_SPLIT = .2
CHUNK_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 500

print("Reading data...")
df = pd.read_csv("dataset.csv")

considered = df[["Month","Day","Hour","Minute","Open","High","Low","Close","Spread","Volume","ema5","ema10","ema25","ema50","ema100","ema200","sma5","sma10","sma25","sma50","sma100","sma200","bands10","bands20","bands40","bands80","bands160","sar1","sar2","sar4","sar8","sar16","sar32","macd6a","macd6b","macd12a","macd12b","macd24a","macd24b","macd48a","macd48b","macd92a","macd92b","macd184a","macd184b","rsi7","rsi14","rsi30","rsi60","rsi120","rsi250","stoch5a","stoch5b","stoch14a","stoch14b","stoch21a","stoch21b","stoch42a","stoch42b","stoch84a","stoch84b","adx7","adx14","adx28","adx52","adx104","adx208","momentum7","momentum14","momentum28","momentum52","momentum104","momentum208","mfi7","mfi14","mfi28","mfi56","mfi112","mfi224","atr5","atr10","atr20","atr40","atr80","atr160"]]
expected = df[["Long","Short","Hold"]]

print("Scaling data...")
considered_data = preprocessing.MinMaxScaler().fit_transform(considered.to_numpy())
expected_data = expected.to_numpy()

def prepare_data(dataset, target, chunk_size):
    size = len(considered_data) - chunk_size

    train_data = []
    train_labels = []

    for i in range(size):
        element = []

        for j in range(chunk_size):
            element.append(considered_data[i + j])

        train_data.append(element)

        if expected_data[i + chunk_size - 1][0] == 1:
            train_labels.append(0)
        elif expected_data[i + chunk_size - 1][1] == 1:
            train_labels.append(1)
        else:
            train_labels.append(2)

    return np.array(train_data), np.array(train_labels)

print("Preparing data...")
train_data, train_labels = prepare_data(considered_data, expected_data, CHUNK_SIZE)
print("Data ready.")


print("Train data shape:", train_data.shape, "type", type(train_data))
print("Train labels shape:", train_labels.shape, "type", type(train_labels))

model = tf.keras.models.load_model("val_model.h5")
tf.keras.backend.set_learning_phase(0)

predicted = model.predict_classes(train_data[-1000:])
print("Predicted:", predicted)
print("Predicted shape:", predicted.shape)

print("Labels:", train_labels)

print(classification_report(train_labels[-1000:], predicted, target_names=["Long", "Short", "Hold"]))
print(confusion_matrix(train_labels[-1000:], predicted))
