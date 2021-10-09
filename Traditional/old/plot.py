from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn import preprocessing

import tensorflow as tf

import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt


PAST_HISTORY = 1
FUTURE_TARGET = 5
STEP = 1
BATCH_SIZE = 1
BUFFER_SIZE = 10000


df = pd.read_csv("dataset.csv")
considered = df[["Month", "Day", "Open", "High", "Low", "MFI", "RSI", "ATR", "EMA"]]
expected = df[["Close"]]

print(considered)
print(expected)
scaler = preprocessing.StandardScaler().fit(considered.values)

data = scaler.transform(considered.values)
print("Data shape: ", data.shape)
scaler2 = preprocessing.StandardScaler().fit(expected.values)

data2 = scaler2.transform(expected.values)
print("Data2 shape: ", data2.shape)

#print(scaler.inverse_transform(data))


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size  # 0 + 1000 / 4000 + 1000

    if end_index is None:
        end_index = len(dataset) - target_size  # 6400

    # 5000

    for i in range(start_index, end_index):  # 1000 - 4000 / 4000 - 6400
        indices = range(i-history_size, i, step)  # (i - 1000) - i
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)


TRAIN_SPLIT = int(len(data)*.7)
print("Train split: ", TRAIN_SPLIT)

def create_time_steps(length):
  return list(range(-length, 0))


def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, history, label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'go',
             label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()



x_val, y_val = multivariate_data(data, data2[:, 0], TRAIN_SPLIT,
                                 None, PAST_HISTORY,
                                 FUTURE_TARGET, STEP,
                                 single_step=False)

val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.batch(BATCH_SIZE).repeat()

model = tf.keras.models.load_model("model.h5")

i = 0
hist = 50

past_price = np.empty([hist])

for x, y in val_data.take(hist + 10):

    if i >= hist:
        multi_step_plot(past_price, y[0], model.predict(x)[0])
    else:
        model.predict(x)
        arr = np.array(x[0])
        #print(arr[0,2])
        past_price[i] = arr[0, 2]

    i += 1
