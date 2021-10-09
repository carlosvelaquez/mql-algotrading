from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn import preprocessing

import os
import time
import zmq

import tensorflow as tf
import numpy as np
import pandas as pd

from io import StringIO

CHUNK_SIZE = 100

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

model = tf.keras.models.load_model("best_model.h5")
tf.keras.backend.set_learning_phase(0)

def predict(csv):
    io = StringIO(csv)
    df = pd.read_csv(io)

    considered = df[["Month","Day","Hour","Minute","Open","High","Low","Close","Spread","Volume","ema5","ema10","ema25","ema50","ema100","ema200","sma5","sma10","sma25","sma50","sma100","sma200","bands10","bands20","bands40","bands80","bands160","sar1","sar2","sar4","sar8","sar16","sar32","macd6a","macd6b","macd12a","macd12b","macd24a","macd24b","macd48a","macd48b","macd92a","macd92b","macd184a","macd184b","rsi7","rsi14","rsi30","rsi60","rsi120","rsi250","stoch5a","stoch5b","stoch14a","stoch14b","stoch21a","stoch21b","stoch42a","stoch42b","stoch84a","stoch84b","adx7","adx14","adx28","adx52","adx104","adx208","momentum7","momentum14","momentum28","momentum52","momentum104","momentum208","mfi7","mfi14","mfi28","mfi56","mfi112","mfi224","atr5","atr10","atr20","atr40","atr80","atr160"]]
    considered_data = preprocessing.MinMaxScaler().fit_transform(considered.to_numpy())
    considered_data = np.resize(considered_data, (1, considered_data.shape[0], considered_data.shape[1]))

    # print(considered_data)
    # print(considered_data.shape)

    predictions = model.predict_classes(considered_data)
    confidences = model.predict(considered_data)

    prediction = predictions[0]
    confidence = confidences[0][2]


    if prediction == 0:
        confidence = confidences[0][0]
    elif prediction == 1:
        confidence = confidences[0][1]

    return str(prediction) + "," + str(confidence) + "," + str(confidences[0][0]) + "," + str(confidences[0][1]) + "," + str(confidences[0][2])

print("Server is up.")

while(True):
    received = socket.recv()
    message = str(received, 'utf-8')
    socket.send(predict(message).encode())
