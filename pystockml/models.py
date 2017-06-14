#!/usr/bin/env python
# -*- coding: utf-8 -*-

'models - machine learning models for the stock prediction problem'

import numpy as np
import pandas as pd

from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError

from statsmodels.tsa.arima_model import ARIMA

from pystockml import statistics

from sklearn.externals import joblib

COLUMNS = r'adj_close sma bandwidth %b momentum volatility adj_volume '\
           'beta'.split()


def build_lstm(input_dim=1, input_length=1, output_dim=1, dropout=.4,
               hidden_size=32, layers=3, loss='mse', optimizer='nadam'):

    if layers < 2:
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

    return model


def build_mlp(input_dim=1, output_dim=1, dropout=.5, hidden_size=64, layers=3,
              loss='mse', optimizer='nadam'):

    model = Sequential()

    model.add(Dense(input_shape=(input_dim,), units=hidden_size))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    for _ in range(layers - 2):
        model.add(Dense(hidden_size))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

    model.add(Dense(units=output_dim))
    model.add(Activation('linear'))

    model.compile(loss=loss, optimizer=optimizer)

    return model


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

    if benchmark_path:
        benchmark = pd.read_csv(benchmark_path)
    elif 'beta' in COLUMNS:
        # Remove beta, which depends on benchmark
        COLUMNS.pop()

    return statistics.augment(df, benchmark)[COLUMNS]


def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df)

    return df, scaler


def train_test_split(df):
    train_size = int(len(df) * .5)

    return df[:train_size], df[train_size:]


def build_dataset(values, shift=1, price_column=0, lookback=0):
    '''Builds a training dataset.

    :param values: The values to read from (dataframe).
    :param shift: How far into the future we want to look.
    :param price_column: The column that contains the adjusted close price.
    :param lookback: How many points from the past we want to include.
    '''
    x, y = [], []

    lines = len(values) - shift
    columns = values.shape[1] if len(values.shape) > 1 else None

    for i in range(lookback, lines - shift):
        # This is *not* an off-by-one error. Assume you have a list of one
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
        y = np.array(y).reshape((-1, columns if price_column == 'all'
                                                or columns is None
                                             else 1))

    return x, y


def build_arima(n_ar_params=3, n_ar_diffs=1, n_ma_params=1, freq='D'):
    return ArimaRegressor(n_ar_params, n_ar_diffs, n_ma_params, freq)


def sma_predictions(X_test):
    sma_column = COLUMNS.index('sma')
    return X_test[:, sma_column]


def grid_search_arima(X, y, params, diffs, ma_params, cv, refit=True):
    best_score, best_configuration, best_model = float('inf'), None, None
    for p in params:
        for d in diffs:
            for m in ma_params:
                arima = build_arima(p, d, m)
                print('Cross-validating ARIMA with order %d %d %d' % (p, d, m))
                for (train_index, test_index) in cv:
                    try:
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        arima.fit(X_train[:, 0], y_train[:, 0])
                        arima_yhat = arima.predict(X_test[:, 0])
                        score = mse(arima_yhat, y_test[:, 0])
                        if score < best_score:
                            best_score = score
                            best_configuration = (p, d, m)
                            best_model = arima
                    except Exception as e:
                        print('Exception ocurred while evaluating model '\
                                '(%d, %d, %d): %s' % (p, d, m, e))
    if best_model and refit:
        best_model.fit(X, y)

    print('Best ARIMA score: {}, configuration: {}'.format(best_score,
                                                           best_configuration))

    return best_score, best_configuration, best_model


def cross_validate_model(model_name, X, y, refit=True, length=1):
    model_name = model_name.lower().strip()
    if model_name not in 'ols ridge huber knn arima lstm'.split():
        raise ValueError('Model %s not supported.' % model_name)

    tscv = TimeSeriesSplit(n_splits=3)
    cv = [(train_index, test_index)
          for train_index, test_index in tscv.split(X)]

    if model_name == 'arima':
        # This is different
        return grid_search_arima(X, y, [1, 3, 5, 10], [0, 1, 2], [0, 1, 5], cv)

    if model_name == 'lstm':
        model = KerasRegressor(build_fn=build_lstm, epochs=25, verbose=1,
                               batch_size=64)
        grid = [
            {
                'input_length': [length], 'dropout': [0.2, 0.5, 0.8],
                'hidden_size': [32, 64], 'input_dim': [X.shape[1]],
                'layers': [2, 3], 'optimizer': 'adam nadam rmsprop'.split(),
            },
        ]
        X = X.reshape(X.shape[0], length, X.shape[1])
    elif model_name == 'ols':
        model = LinearRegression()
        grid = [{'normalize': [True, False], 'fit_intercept': [True, False],
                 'n_jobs': [-1]}]
    elif model_name == 'ridge':
        model = Ridge()
        grid = [{'alpha': [1.0, 10.0, 0.1, 0.01], 'normalize': [True, False],
                 'fit_intercept': [True, False]}]
    elif model_name == 'huber':
        model = HuberRegressor()
        grid = [{'epsilon': [1.1, 1.35, 1.5], 'max_iter': [10, 100, 1000],
                 'fit_intercept': [True, False]}]
    else:
        # knn
        model = KNeighborsRegressor()
        grid = [{'n_neighbors': [1, 3, 5, 10], 'weights': ['uniform', 'distance'],
                 'p': [1, 2], 'n_jobs': [-1]}]
    gs = GridSearchCV(estimator=model, param_grid=grid,
                      n_jobs=-1 if model_name != 'lstm' else 1,
                      cv=cv, verbose=0)
    gs.fit(X, y)
    return gs.best_score_, gs.best_params_, gs.best_estimator_


def get_preprocessed_datasets(tickers, train_size=.8, shift=.1, lookback=0,
                              lstm=False):

    dfs = []
    scaler = MinMaxScaler(feature_range=(0, 1))

    for ticker in tickers:
        try:
            df = load_data('data/%s.csv.gz' % ticker).fillna(method='bfill')
            scaler.partial_fit(df)
            dfs.append(df)
        except Exception as e:
            print('Ignoring {} because I failed to read it: {}'
                  .format(ticker,e))
            continue

    dfs = [scaler.transform(df) for df in dfs]
    ret = {}

    for ticker, df in zip(tickers, dfs):
        X, y = build_dataset(df, shift, COLUMNS.index('adj_close'), lookback)
        cut_point = int(train_size * X.shape[0])
        X_train = X[:cut_point]
        y_train = y[:cut_point]
        X_test = X[cut_point:]
        y_test = y[cut_point:]

        if lstm and not lookback:
            X_train = X_train.reshape(
                X_train.shape[0],
                lookback + 1,
                X_train.shape[1]
            )

            X_test = X_test.reshape(
                X_test.shape[0],
                lookback + 1,
                X_test.shape[1]
            )

        ret[ticker] = X_train, y_train, X_test, y_test

    return ret, scaler


def get_processed_dataset(ticker, train_size=0.8, shift=1, lookback=0,
                          lstm=False):
    df = load_data('data/%s.csv.gz' % ticker)
    df, scaler = preprocess_data(df.values)

    X, y = build_dataset(df, shift, 'all', lookback)
    cut_point = int(train_size * X.shape[0])
    X_train = X[:cut_point]
    y_train = y[:cut_point]
    X_test = X[cut_point:]
    y_test = y[cut_point:]

    return X_train, y_train, X_test, y_test, scaler


def main():
    np.random.seed(1234)

    import seaborn as sns
    import matplotlib.pyplot as plt

    for ticker in 'AAPL AIR BA FDX IBM MSFT T TSLA'.split():
        for shift in [1, 5, 15, 21]:
            X_train, y_train, X_test, y_test, scaler = get_processed_dataset(
                ticker, .8, shift
            )
            y_train = y_train[:, 0].reshape(-1, 1)
            y_test = y_test[:, 0].reshape(-1, 1)

            for model_name in 'ols ridge huber knn arima lstm'.split():
                print('ticker: {}, model: {}, shift: {}'.format(
                    ticker, model_name, shift)
                )
                score, params, estimator = cross_validate_model(
                    model_name,
                    X_train,
                    y_train,
                )
                yhat = estimator.predict(X_test)
                model = {
                    'score': score,
                    'params': params,
                    'yhat': yhat,
                    'mse': mse(yhat, y_test),
                    'r2': r2_score(yhat, y_test),
                    'y_test': y_test,
                    'scaler': scaler,
                }
                if model_name != 'arima':
                    model['estimator'] = estimator
                joblib.dump(
                    model,
                    'models/{}-{}-{}.pkl'.format(model_name, ticker, shift)
                )


if __name__ == '__main__':
    main()
