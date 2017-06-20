#!/usr/bin/env python
# -*- coding: utf-8 -*-

'Predicts stock prices when called from the command line.'

import warnings
warnings.filterwarnings("ignore")

import os
import argparse

import numpy as np
import pandas as pd

from pystockml import models  # pylint: disable=wrong-import-position

def main():
    'Main function.'
    try:
        tickers = os.listdir('data')
        tickers = (os.path.basename(t) for t in tickers if t.endswith('.csv.gz'))
        tickers = [t.split('.')[0] for t in tickers]
        epilog = '\nValid values for ticker are: %s' % tickers
        epilog += '\n\nSample usage: ./predict.py IBM 2010-01-01 2011-01-01 21'
    except OSError:
        tickers = []
        epilog = ''

    parser = argparse.ArgumentParser(description='Predicts stock prices.',
                                     epilog=epilog, add_help=True)
    parser.add_argument('ticker', metavar='TICKER',
                        help='The stock item to predict')
    parser.add_argument('start_date', metavar='START_DATE',
                        help='The initial date to start looking into history.')
    parser.add_argument('end_date', metavar='END_DATE',
                        help='The final date to stop looking into history.')
    parser.add_argument('shift', metavar='SHIFT', type=int,
                        help='How many days in advance to predict.')

    options = parser.parse_args()

    if not tickers:
        print('"No tickers available. Unable to predict.')
        raise SystemExit

    ticker = options.ticker
    if ticker not in tickers:
        print('"ticker" must be one of %s' % tickers)
        raise SystemExit

    try:
        start_date = pd.to_datetime(options.start_date)
    except ValueError:
        print('"start_date" must be a valid date. Not %s' % start_date)
        raise SystemExit

    try:
        end_date = pd.to_datetime(options.end_date)
    except ValueError:
        print('"end_date" must be a valid date. Not %s' % end_date)
        raise SystemExit

    shift = options.shift
    if shift <= 0:
        print('"shift" must be a positive integer')
        raise SystemExit

    print('Loading data...')
    X, y, tX, _, scaler = models.get_processed_dataset(
        ticker, 0.9999999999, shift, 0, False, start_date, end_date
    )

    print('Training model...')
    _, _, model = models.cross_validate_model('huber', X, y)

    print('Predicting...')
    yhat = model.predict(tX)

    prediction = scaler.inverse_transform(
        np.array([[yhat[0]] + [0] * (X.shape[1] - 1)])
    )

    print('Predicted value:', prediction[0, 0])

if __name__ == '__main__':
    main()
