from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn import preprocessing
from tensorflow.keras import models, layers, optimizers, metrics, callbacks

import os
import datetime

import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.mixed_precision import experimental as mixed_precision

VAL_SPLIT = .2
CHUNK_SIZE = 85
BATCH_SIZE = 32
EPOCHS = 500
HIDDEN_SIZE = 32
DROPOUT = 0.2

print("Reading data...")
df = pd.read_csv("dataset.csv")

considered = df[["Month","Day","Hour","Minute","Open","High","Low","Close","Spread","Volume","ema5","ema10","ema25","ema50","ema100","ema200","sma5","sma10","sma25","sma50","sma100","sma200","bands10","bands20","bands40","bands80","bands160","sar1","sar2","sar4","sar8","sar16","sar32","macd6a","macd6b","macd12a","macd12b","macd24a","macd24b","macd48a","macd48b","macd92a","macd92b","macd184a","macd184b","rsi7","rsi14","rsi30","rsi60","rsi120","rsi250","stoch5a","stoch5b","stoch14a","stoch14b","stoch21a","stoch21b","stoch42a","stoch42b","stoch84a","stoch84b","adx7","adx14","adx28","adx52","adx104","adx208","momentum7","momentum14","momentum28","momentum52","momentum104","momentum208","mfi7","mfi14","mfi28","mfi56","mfi112","mfi224","atr5","atr10","atr20","atr40","atr80","atr160"]]
#considered = df[["ema10","ema25","ema50","ema100","ema200","sma10","sma25","sma50","sma100","sma200","bands10","bands20","bands40","bands80","bands160","sar2","sar4","sar8","sar16","sar32","macd12a","macd12b","macd24a","macd24b","macd48a","macd48b","macd92a","macd92b","macd184a","macd184b","rsi14","rsi30","rsi60","rsi120","rsi250","stoch14a","stoch14b","stoch21a","stoch21b","stoch42a","stoch42b","stoch84a","stoch84b","adx14","adx28","adx52","adx104","adx208","momentum7","momentum14","momentum28","momentum52","momentum104","momentum208","mfi7","mfi14","mfi28","mfi56","mfi112","mfi224","atr10","atr20","atr40","atr80","atr160"]]
#expected = df[["Long","Short","Hold"]]
expected = df[["Close"]]

print("Scaling data...")
considered_data = preprocessing.MinMaxScaler().fit_transform(considered.to_numpy())
expected_data = expected.to_numpy()

cold = []

def prepare_data(dataset, target, chunk_size):
    size = len(dataset) - chunk_size - 1
    hold_count = 0
    long_count = 0
    short_count = 0

    train_data = []
    train_labels = []
    sample_weights = []

    for i in range(size):
        element = []

        for j in range(chunk_size):
            element.append(considered_data[i + j])

        # exp = expected_data[i + chunk_size - 1]
        exp = expected_data[i + chunk_size]
            
        # if exp[0] == 1:
        #     cold.append(0)
        #     sample_weights.append(9.940422152)
        #     long_count += 1
        # elif exp[1] == 1:
        #     cold.append(1)
        #     sample_weights.append(10.02231121)
        #     short_count += 1
        # else:
        #     cold.append(2)
        #     sample_weights.append(1.250588924)
        #     hold_count += 1

        train_data.append(element)
        train_labels.append(exp)

    #print("Long: %i, Short: %i, Hold (considered): %i" % (long_count, short_count, hold_count))
    return np.array(train_data), np.array(train_labels), np.array(sample_weights)



print("Preparing data...")
train_data, train_labels, sample_weights = prepare_data(considered_data, expected_data, CHUNK_SIZE)
print("Data ready.")


print("Train data shape:", train_data.shape, "type", type(train_data))
print("Train labels shape:", train_labels.shape, "type", type(train_labels))

model = models.Sequential()

model.add(layers.LSTM(HIDDEN_SIZE, activation='relu', return_sequences=True, input_shape=train_data.shape[-2:]))
model.add(layers.LSTM(HIDDEN_SIZE, activation='relu', return_sequences=True))
model.add(layers.LSTM(HIDDEN_SIZE, activation='relu'))
model.add(layers.Dense(1, activation='relu'))

opt = optimizers.Adam()
#opt = optimizers.SGD(momentum=0.8)
opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

#model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[metrics.CategoricalAccuracy(), metrics.Precision(), metrics.Recall()])
model.compile(optimizer=opt, loss='mse', metrics=[metrics.Accuracy(), metrics.Precision(), metrics.Recall()])

es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, min_delta=0.0001)
#rlp = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=20, verbose=1, mode='min', min_delta=0.001, cooldown=1, min_lr=0.0001)
mcp = callbacks.ModelCheckpoint("best_model.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

#history = model.fit(x=train_data, y=train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VAL_SPLIT, shuffle=True, callbacks=[mcp, es, tb], sample_weight=sample_weights)
history = model.fit(x=train_data, y=train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VAL_SPLIT, shuffle=True, callbacks=[mcp, es, tb])

model.save("last_model.h5")
