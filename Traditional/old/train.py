from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn import preprocessing

import tensorflow as tf

import numpy as np
import os
import pandas as pd

from tensorflow.keras.mixed_precision import experimental as mixed_precision

PAST_HISTORY = 1
FUTURE_TARGET = 5
STEP = 1
BATCH_SIZE = 1
BUFFER_SIZE = 10000

EPOCHS = 10
HIDDEN_RATIO = .66



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

x_train, y_train = multivariate_data(data, data2[:, 0], 0,
                                     TRAIN_SPLIT, PAST_HISTORY,
                                     FUTURE_TARGET, STEP,
                                     single_step=False)

x_val, y_val = multivariate_data(data, data2[:, 0], TRAIN_SPLIT,
                                 None, PAST_HISTORY,
                                 FUTURE_TARGET, STEP,
                                 single_step=False)

# print ('Single window of past history : {}'.format(x_train[0].shape))
# print('Target temperature to predict : {}'.format(y_train[0].shape))

print(x_train)
print(x_train.shape)
print(x_train.shape[-2:])

print(y_train)
print(y_train.shape)
print(y_train.shape[-2:])

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_data = train_data.cache().batch(BATCH_SIZE).repeat()

val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.batch(BATCH_SIZE).repeat()


def create_time_steps(length):
  return list(range(-length, 0))


#HIDDEN_SIZE = int(len(x_train[1])*len(x_train[2])*HIDDEN_RATIO)
HIDDEN_SIZE = 16

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, input_shape=x_train.shape[-2:]))
#model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(int(HIDDEN_SIZE), activation="relu"))
#model.add(tf.keras.layers.Dropout(0.2))
#model.add(tf.keras.layers.LSTM(int(HIDDEN_SIZE)))
#model.add(tf.keras.layers.Dropout(0.2))
#model.add(tf.keras.layers.Dense(HIDDEN_SIZE, activation='relu'))
model.add(tf.keras.layers.Dense(FUTURE_TARGET))

opt = tf.keras.optimizers.Adam()
opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
model.compile(optimizer=opt, loss='mse')


EVALUATION_INTERVAL = int(len(data)/BATCH_SIZE)

history = model.fit(train_data, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL,
                    validation_data=val_data, validation_steps=50)

model.save("model.h5")
