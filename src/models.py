from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adagrad
import numpy as np

def LSTM_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=1500, input_shape=input_shape))
    model.add(Dense(units=5))
    model.compile(optimizer=Adagrad(), loss='mse')

    return model