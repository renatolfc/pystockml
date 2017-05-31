#!/usr/bin/env python
# -*- coding: utf-8 -*-

'statistics - statistic functions useful for financial data analysis.'

import numpy as np
import pandas as pd

TRADING_DAYS_IN_YEAR = 252


def rolling_mean(values, window=21):
    'Computes rolling mean of given values, using a window of size `window`.'

    if isinstance(values, pd.Series) or isinstance(values, pd.DataFrame):
        return values.rolling(window=window, center=False).mean()

    return pd.rolling_mean(values, window=window)


def rolling_std(values, window=21):
    'Computes rolling std of given values, using a window of size `window`.'

    if isinstance(values, pd.Series) or isinstance(values, pd.DataFrame):
        return values.rolling(window=window, center=False).std()

    return pd.rolling_std(values, window=window)


def percent_bollinger(values, rm, rstd):
    'Computes [%b](https://en.wikipedia.org/wiki/Bollinger_Bands).'

    lower, upper = bollinger_bands(rm, rstd)

    if not isinstance(values, np.ndarray):
        values = values.values

    return ((np.hstack((np.array([float('nan')]), values[1:])) - lower) /
            (upper - lower))


def bollinger_bands(rm, rstd):
    'Computes bollinger bands given rolling mean and rolling std deviation.'
    upper = rm + rstd * 2
    lower = rm - rstd * 2

    return lower, upper


def bandwidth(rm, rstd):
    'Computes bollinger bandwidth.'
    lower, upper = bollinger_bands(rm, rstd)

    return (upper - lower) / rm


def historical_volatility(values):
    'Returns the annualized stddev of daily log returns of values.'

    if not isinstance(values, np.ndarray):
        values = values.values

    returns = np.log(values[1:] / values[:-1])

    return np.sqrt(TRADING_DAYS_IN_YEAR * returns.var())


def volatility(values, period=TRADING_DAYS_IN_YEAR):
    'Calculates the volatility in a fixed period.'

    if not isinstance(values, np.ndarray):
        values = values.values

    head = np.array([float('nan')] * period)

    tail = np.array([
        historical_volatility(values[i:i + period])
        for i in range(len(values) - period)
    ])

    return np.hstack((head, tail))


def historical_beta(values, benchmark):
    'Computes the financial beta between a stock and a benchmark.'

    cov = np.cov(np.vstack((values, benchmark)))

    return (cov / benchmark.var())[0, 1]


def beta(values, benchmark, period=TRADING_DAYS_IN_YEAR):
    'Rolling beta. Like `volatility`.'

    if not isinstance(values, np.ndarray):
        values = values.values
    if not isinstance(benchmark, np.ndarray):
        benchmark = benchmark.values

    head = np.array([float('nan')] * period)

    tail = np.array([
        historical_beta(values[i:i + period], benchmark[i:i + period])
        for i in range(len(values) - period)
    ])

    return np.hstack((head, tail))


def exponential_rolling_mean(values):
    'Computes the exponential rolling mean of values.'

    return pd.ewma(values)


def momentum(values, window=21):
    'Computes the momentum of values.'

    if not isinstance(values, np.ndarray):
        values = values.values

    beginning = np.array([float('nan')] * window)

    return np.hstack((beginning, values[window:] / values[:-window] - 1))


def augment(dataframe, benchmark=None, column='adj_close', window=21):
    'Augments a dataframe with the statistics found in this module.'

    rm = rolling_mean(dataframe[column], window)
    rstd = rolling_std(dataframe[column], window)
    bw = bandwidth(rm, rstd)
    pb = percent_bollinger(dataframe[column], rm, rstd)
    dv = momentum(dataframe[column], window)
    v = volatility(dataframe[column])

    dataframe['sma'] = rm
    dataframe['bandwidth'] = bw
    dataframe[r'%b'] = pb
    dataframe['momentum'] = dv
    dataframe['volatility'] = v

    if benchmark is not None:
        dataframe['beta'] = beta(dataframe[column], benchmark[column])

    return dataframe.ix[TRADING_DAYS_IN_YEAR:]
