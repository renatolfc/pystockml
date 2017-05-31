#!/usr/bin/env python
# -*- coding: utf-8 -*-

'models - machine learning models for the stock prediction problem'

import numpy as np
import pandas as pd

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse

from pystockml import statistics

COLUMNS = r'adj_close sma bandwidth %b momentum volatility beta'.split()


def load_data(path, benchmark_path=None):
    benchmark = None
    df = pd.read_csv(path)
    df.index = df['date']

    columns = COLUMNS

    if benchmark_path:
        benchmark = pd.read_csv(benchmark_path)
    else:
        # Remove beta, which depends on benchmark
        columns.pop()

    return statistics.augment(df, benchmark)[COLUMNS]


def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df)

    return df, scaler


def train_test_split(df):
    train_size = int(len(df) * .5)

    return df[:train_size], df[train_size:]


def build_dataset(values, shift=1, column=0):
    ':param shift: How far into the future we want to look.'
    x, y = [], []
    for i in range(values.shape[0] - shift - 1):
        x.append(values[i:(i + shift), column])
        y.append(values[i + shift, column])
    return np.array(x), np.array(y)


def build_lstm(input_dim=1, output_dim=1, dropout=0.4, hidden_size=32,
               layers=3):

    model = Sequential()

    model.add(LSTM(input_shape=(None, input_dim), units=hidden_size,
                   return_sequences=True))
    model.add(Dropout(dropout))

    for _ in range(layers - 2):
        model.add(LSTM(hidden_size, return_sequences=True))
        model.add(Dropout(dropout))

    model.add(LSTM(hidden_size, return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(units=output_dim))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')

    return model


if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = load_data('data/MSFT.csv')
    df, scaler = preprocess_data(df['adj_close'].values)

    train, test = train_test_split(df.reshape((-1, 1)))
    train_X, train_y = build_dataset(train)
    test_X, test_y = build_dataset(test)

    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

    lstm = build_lstm()
    lstm.fit(train_X, train_y, epochs=200, batch_size=256, verbose=1)

    yhat = lstm.predict(test_X)
    score = mse(test_y.reshape(-1, 1), yhat.reshape(-1, 1))

    true_test_y = scaler.inverse_transform(test_y)
    true_yhat = scaler.inverse_transform(yhat)

    print('Mean Squared Error:', score)

    plt.plot(true_yhat)
    plt.plot(true_test_y)

    plt.legend(('Predicted', 'Actual'))

    plt.show()
