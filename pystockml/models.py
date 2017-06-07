#!/usr/bin/env python
# -*- coding: utf-8 -*-

'models - machine learning models for the stock prediction problem'

import numpy as np
import pandas as pd

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError

from statsmodels.tsa.arima_model import ARIMA

from pystockml import statistics

COLUMNS = r'adj_close sma bandwidth %b momentum volatility beta'.split()


class LstmRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=1, input_length=1, output_dim=1, dropout=.4,
                 hidden_size=32, layers=3, loss='mse', optimizer='nadam'):

        if layers <= 2:
            raise ValueError('LstmRegressor must have at least two layers.')

        model = Sequential()

        model.add(LSTM(input_shape=(input_length, input_dim), units=hidden_size,
                       return_sequences=True))
        model.add(Dropout(dropout))

        for _ in range(layers - 2):
            model.add(LSTM(hidden_size, return_sequences=True))
            model.add(Dropout(dropout))

        model.add(LSTM(hidden_size, return_sequences=False))
        model.add(Dropout(dropout))

        model.add(Dense(units=output_dim))
        model.add(Activation('linear'))

        model.compile(loss=loss, optimizer=optimizer)

        self.model = model


    def fit(self, X, y, epochs=200, batch_size=256, verbose=0):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size,
                       verbose=verbose)
        return self


    def predict(self, X):
        return self.model.predict(X)


class ArimaRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_ar_params=3, n_ar_diffs=1, n_ma_params=1, freq='D'):
        '''Builds an ARIMA regressor.

        :param n_ar_params: The number of autoregressive parameters.
        :param n_ar_diffs: The number of autoregressive differences.
        :param n_ma_params: The number of moving average parameters.
        :param freq: The frequency of the time series. Defaults to 'D' (day).
        '''

        self.n_ar_params = n_ar_params
        self.n_ar_diffs = n_ar_diffs
        self.n_ma_params = n_ma_params

        self.model, self.model_fit = None, None

    def fit(self, X, y):
        '''Fit model.

        :param X: The input time series.
        :param y: The actual values.

        This method also stores the provided X and y parameters, because we
        need them to retrain the ARIMA model every time we do a new prediction.
        '''

        self.y_train_ = [y_ for y_ in y]

        self._fit()

        return self

    def _fit(self):
        'Updates the model using the stored X and y arrays.'

        self.model = ARIMA(
            self.y_train_,
            order=(self.n_ar_params, self.n_ar_diffs, self.n_ma_params)
        )
        self.model_fit = self.model.fit(disp=0)

        return self

    def predict(self, X, refit=False):
        'Makes forecasts. If `refit` is False, only uses len(X).'

        if not self.model_fit:
            raise NotFittedError("This %(name)s instance is not fitted "
                                 "yet" % {'name': type(self).__name__})

        if not refit:
            return self.model_fit.forecast(len(X))[0]

        try:
            len(X)
        except TypeError:
            # X is a scalar
            yhat = self.model_fit.forecast()[0]
            self.y_train_.append(X)
            self._fit()
            return yhat

        yhat = []
        for x in X:
            yhat.append(self.model_fit.forecast()[0])
            self.y_train_.append(x)
            self._fit()

        return np.array(yhat)


    def summary(self):
        'Returns the underlying ARIMA model summary.'
        if self.model_fit:
            return self.model_fit.summary()
        return None


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


def build_dataset(values, shift=1, price_column=0, lookback=0):
    ':param shift: How far into the future we want to look.'
    x, y = [], []

    lines = len(values) - shift
    columns = values.shape[1] if len(values.shape) > 1 else None

    for i in range(lookback, lines - shift):
        # This is *not* and off-by-one error. Assume you have a list of one
        # element, that i=0 and lookback=0:
        # >>> a = [1]
        # >>> a[0-0:0+1]
        # >>> [1]
        x.append(values[i-lookback:i+1])
        if price_column == 'all' or columns is None:
            y.append(values[i+shift])
        else:
            y.append(values[i+shift, price_column])

    if lookback:
        x = np.array(x)
        y = np.array(y)
    else:
        x = np.array(x).reshape((-1, columns if columns else 1))
        y = np.array(y).reshape((-1, columns if columns else 1))

    return x, y


def build_lstm(input_dim=1, input_length=1, output_dim=1, dropout=0.4,
               hidden_size=32, layers=3, loss='mse', optimizer='nadam'):

    return LstmRegressor(
        input_dim, input_length, output_dim, dropout, hidden_size, layers,
        loss, optimizer
    )


def build_arima(n_ar_params=3, n_ar_diffs=1, n_ma_params=1, freq='D'):
    return ArimaRegressor(n_ar_params, n_ar_diffs, n_ma_params, freq)


def build_linear_regressor(normalize=False):
    return LinearRegression(normalize=normalize, n_jobs=-1)


def sma_predictions(X_test):
    sma_column = COLUMNS.index('sma')
    return X_test[:, sma_column]


def build_ridge_regressor(normalize=False):
    return Ridge(normalize=normalize)


def build_huber_regressor():
    return HuberRegressor()


def main():
    np.random.seed(1234)

    import seaborn as sns
    import matplotlib.pyplot as plt

    ticker = 'IBM'
    df = load_data('data/%s.csv.gz' % ticker)
    df, scaler = preprocess_data(df.values)

    shift = 1
    lookback = 0
    X, y = build_dataset(df, shift, 'all', lookback)
    tscv = TimeSeriesSplit(n_splits=3)

    arima = ArimaRegressor(5, 1, 1)
    huber = build_huber_regressor()
    ridge = build_ridge_regressor()
    linear = build_linear_regressor()
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        try:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            sma_yhat = sma_predictions(X_test)

            if lookback:
                arima.fit(X_train[:, -1, 0], y_train[:, 0])
                arima_yhat = arima.predict(X_test[:, -1, 0].reshape((-1, 1)),
                                           refit=False)
            else:
                arima.fit(X_train[:, 0], y_train[:, 0])
                arima_yhat = arima.predict(X_test[:, 0], refit=False)

            if lookback:
                linear.fit(X_train[:, -1, 0].reshape((-1, 1)),
                           y_train[:, 0].reshape((-1, 1)))
                yhat = linear.predict(X_test[:, -1, 0].reshape((-1, 1)))
            else:
                linear.fit(X_train, y_train[:, 0])
                yhat = linear.predict(X_test)

            if lookback:
                ridge.fit(X_train[:, -1, 0].reshape((-1, 1)),
                          y_train[:, 0].reshape((-1, 1)))
                ridge_yhat = ridge.predict(X_test[:, -1, 0].reshape((-1, 1)))
            else:
                ridge.fit(X_train, y_train[:, 0])
                ridge_yhat = ridge.predict(X_test)

            if lookback:
                huber.fit(X_train[:, -1, 0].reshape((-1, 1)),
                          y_train[:, 0].reshape((-1, 1)))
                huber_yhat = huber.predict(X_test[:, -1, 0].reshape((-1, 1)))
            else:
                huber.fit(X_train, y_train[:, 0])
                huber_yhat = huber.predict(X_test)

            if lookback:
                X_train_lstm = X_train[:, :, 0].reshape((-1, lookback, 1))
                X_test_lstm = X_test[:, :, 0].reshape((-1, lookback, 1))
            else:
                X_train_lstm = X_train[:, 0].reshape((-1, 1))
                X_train_lstm = X_train_lstm.reshape(
                    (X_train_lstm.shape[0],
                     1 if not lookback else lookback,
                     X_train_lstm.shape[-1])
                )

                X_test_lstm = X_test[:, 0].reshape((-1, 1))
                X_test_lstm = X_test_lstm.reshape(
                    (X_test_lstm.shape[0],
                     1 if not lookback else lookback,
                     X_test_lstm.shape[-1])
                )

            lstm = build_lstm(input_length=lookback if lookback else 1)
            lstm.fit(X_train_lstm, y_train[:, 0], epochs=200, batch_size=256,
                     verbose=1)
            lstm_yhat = lstm.predict(X_test_lstm)

            true_y_test = scaler.inverse_transform(y_test)[:, 0]
            tmp = y_test.copy()
            tmp[:, 0] = yhat.reshape((-1,))
            true_yhat = scaler.inverse_transform(tmp)[:, 0]
            tmp[:, 0] = lstm_yhat.reshape((-1,))
            true_lstm_yhat = scaler.inverse_transform(tmp)[:, 0]
            tmp[:, 0] = arima_yhat.reshape((-1,))
            true_arima_yhat = scaler.inverse_transform(tmp)[:, 0]
            tmp[:, 0] = ridge_yhat.reshape((-1,))
            true_ridge_yhat = scaler.inverse_transform(tmp)[:, 0]
            tmp[:, 0] = huber_yhat.reshape((-1,))
            true_huber_yhat = scaler.inverse_transform(tmp)[:, 0]

            tmp[:, 0] = sma_yhat
            true_sma_yhat = scaler.inverse_transform(tmp)[:, 0]

            score = r2_score(true_sma_yhat, true_y_test)
            print('Benchmark R2 Score:', score)

            score = r2_score(true_yhat, true_y_test)
            print('Linear R2 Score:', score)

            score = r2_score(true_lstm_yhat, true_y_test)
            print('LSTM R2 Score:', score)

            score = r2_score(true_arima_yhat, true_y_test)
            print('ARIMA R2 Score:', score)

            score = r2_score(true_ridge_yhat, true_y_test)
            print('ridge R2 Score:', score)

            score = r2_score(true_huber_yhat, true_y_test)
            print('huber R2 Score:', score)

            fig = plt.figure()

            plt.plot(true_sma_yhat)
            plt.plot(true_yhat)
            plt.plot(true_lstm_yhat)
            plt.plot(true_arima_yhat)
            plt.plot(true_ridge_yhat)
            plt.plot(true_huber_yhat)
            plt.plot(true_y_test)

            plt.legend(('Benchmark', 'Linear Regression', 'LSTM', 'ARIMA', 'Actual'))
            fig.savefig('%s-%d.pdf' % (ticker, i))

        except Exception as e:
            raise
            print('Found exception %s. Continuing...' % e)


if __name__ == '__main__':
    main()
