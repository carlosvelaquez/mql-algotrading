from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn import preprocessing

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

df = pd.read_csv("dataset.csv")
considered = df[["Month", "Day", "Open", "High", "Low", "Close", "MFI", "RSI", "ATR", "EMA"]]

print(considered)

scaler = preprocessing.MinMaxScaler().fit(considered.values)
data = scaler.transform(considered.values)
print("Data shape: ", data.shape)

#print(scaler.inverse_transform(data))


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size # 0 + 1000 / 4000 + 1000

    if end_index is None:
        end_index = len(dataset) - target_size # 6400

    # 5000

    for i in range(start_index, end_index): # 1000 - 4000 / 4000 - 6400
        indices = range(i-history_size, i, step) # (i - 1000) - i
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

TRAIN_SPLIT = int(len(data)*.7)
print("Train split: ", TRAIN_SPLIT)

past_history = 50
future_target = 3
STEP = 1



x_train, y_train = multivariate_data(data, data[:, 5], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=False)

x_val, y_val = multivariate_data(data, data[:, 5], TRAIN_SPLIT,
                                                   None, past_history,
                                                   future_target, STEP,
                                                   single_step=False)

# print ('Single window of past history : {}'.format(x_train[0].shape))
# print('Target temperature to predict : {}'.format(y_train[0].shape))

print(x_train)
print(x_train.shape)
print(x_train.shape[-2:])

BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.batch(BATCH_SIZE).repeat()


def create_time_steps(length):
  return list(range(-length, 0))


def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 5]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'go',
            label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()

HIDDEN_RATIO = .7
HIDDEN_SIZE = int(len(x_train[1])*len(x_train[2])*HIDDEN_RATIO)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.LSTM(HIDDEN_SIZE, activation='relu', return_sequences=True, input_shape=x_train.shape[-2:]))
model.add(tf.keras.layers.LSTM(HIDDEN_SIZE, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.2))
#model.add(tf.keras.layers.LSTM(HIDDEN_SIZE, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(HIDDEN_SIZE, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(future_target))

model.compile(optimizer='adam', loss='mse')


EVALUATION_INTERVAL = int(len(data)/BATCH_SIZE)
EPOCHS = 10

history = model.fit(train_data, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_data, validation_steps=50)


def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()


plot_train_history(history,
                   'Training and validation loss')

for x, y in val_data.take(2):
  multi_step_plot(x[0], y[0], model.predict(x)[0])

# mean = dataset[:TRAIN_SPLIT].mean()
# std = dataset[:TRAIN_SPLIT].std()

# print(mean)
# print(std)

# dataset = (dataset - mean)/std
# print(dataset)
