#!/usr/bin/env python
# -*- coding: utf-8 -*-

'statistics - statistic functions useful for financial data analysis.'

import numpy as np
import pandas as pd

TRADING_DAYS_IN_YEAR = 252


def rolling_mean(values, window):
    'Computes rolling mean of given values, using a window of size `window`.'
    return pd.rolling_mean(values, window=window)


def rolling_std(values, window):
    'Computes rolling std of given values, using a window of size `window`.'
    return pd.rolling_std(values, window=window)


def bollinger_bands(rm, rstd):
    'Computes bollinger bands given rolling mean and rolling std deviation.'
    upper = rm + rstd * 2
    lower = rm - rstd * 2

    return lower, upper


def historical_volatility(values):
    'Returns the annualized stddev of daily log returns of values.'

    if not isinstance(values, np.array):
        values = values.values

    returns = np.log(values[1:] / values[:-1])

    return np.sqrt(TRADING_DAYS_IN_YEAR * returns.var())


def beta(values, benchmark):
    'Computes the financial beta between a stock and a benchmark.'

    cov = np.cov(np.vstack((values, benchmark)).T)

    return cov / benchmark.var()

def exponential_rolling_mean(values):
    'Computes the exponential rolling mean of values.'

    return pd.ewma(values)


def momentum(values, window):
    'Computes the momentum of values.'

    return values[window:] / values[:-window] - 1
