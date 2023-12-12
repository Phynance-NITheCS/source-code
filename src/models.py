from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import *
import numpy as np

def LSTM_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=500, input_shape=input_shape))
    model.add(Dense(units=5))
    model.compile(optimizer=SGD(), loss='mse')

    return model