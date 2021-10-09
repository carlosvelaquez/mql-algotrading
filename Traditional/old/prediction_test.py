from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn import preprocessing

import os

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRAIN_SPLIT = .9
CHUNK_SIZE = 100
BATCH_SIZE = 32

# df = pd.read_csv("dataset2.csv")
df = pd.read_csv("dataset_15m_5y.csv")

considered = df[["Open","High","Low","ema5","ema10","ema25","ema50","ema100","ema200","sma5","sma10","sma25","sma50","sma100","sma200","rsi7","rsi14","rsi30","rsi60","rsi120","rsi250","sar1","sar2","sar4","sar8","sar16","sar32","stoch5a","stoch14a","stoch21a","stoch42a","stoch84a","stoch5b","stoch14b","stoch21b","stoch42b","stoch84b","macd6a","macd12a","macd24a","macd48a","macd92a","macd184a","macd6b","macd12b","macd24b","macd48b","macd92b","macd184b","adx7","adx14","adx28","adx52","adx104","adx208","momentum7","momentum14","momentum28","momentum52","momentum104","momentum208","mfi7","mfi14","mfi28","mfi56","mfi112","mfi224","atr5","atr10","atr20","atr40","atr80","atr160"]]
# expected = df[["Long","CloseLong","Short","CloseShort"]]
expected = df[["Long","Short","Hold"]]

consideredScaler = preprocessing.MinMaxScaler().fit(considered.values)
# consideredScaler = preprocessing.PowerTransformer().fit(considered.values)
# expectedScaler = preprocessing.MinMaxScaler().fit(expected.values)

considered_data = consideredScaler.transform(considered.values)
# expected_data = expectedScaler.transform(expected.values)
expected_data = expected.values

# print(considered_data)
# print(expected_data)

def prepare_data(dataset, target, chunk_size):
    size = len(considered_data) - chunk_size

    train_data = []
    train_labels = []

    val_data = []
    val_labels = []

    

    for i in range(int(size*TRAIN_SPLIT), size):
        element = []

        for j in range(chunk_size):
            element.append(considered_data[i + j])

        if i >= size*TRAIN_SPLIT:
            val_data.append(element)
            val_labels.append(expected_data[i])

    val_shape = (len(val_data), len(val_data[0]), len(val_data[0][0]), 1)
    print(val_shape)

    return np.array(val_data).reshape(val_shape), np.array(val_labels)

x_val, y_val = prepare_data(considered_data, expected_data, CHUNK_SIZE)

global point
point = 1

# def onclick(event):
#     global point
    
#     event.canvas.figure.clear()
#     event.canvas.figure.gca().imshow(x_train[point])
#     point += 1

#     event.canvas.draw()


# fig = plt.figure()
# fig.canvas.mpl_connect('button_press_event', onclick)
# imshow = plt.imshow(x_train[0])
# plt.show()

# imshow = plt.imshow(considered_data[:250])
# plt.show()

model = tf.keras.models.load_model("best_model.h5")

predictions = model.predict_classes(x_val)
confidences = model.predict(x_val)
correct = 0

start = 0
count = 100

for i in range(start, start + count):
    prediction = predictions[i]
    expected = 2
    confidence = np.std(confidences)

    if y_val[i][0] == 1:
        expected = 0
    elif y_val[i][1] == 1:
        expected = 1

    if expected == prediction:
        correct += 1
        print("Confidence:", confidence, "| Predicted:", predictions[i], "| Expected:", expected, "| Acc:", (correct/((i - start) + 1))*100)
    else:
        print("Confidence:", confidence, "| Predicted:", predictions[i], "| Expected:", expected, "| Acc:", (correct/((i - start) + 1))*100, "(wrong)")

