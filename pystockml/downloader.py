#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


def download(quandl, ticker, start_date, end_date):
    date = {
        'gte': start_date,
        'lte': end_date,
    }
    df = quandl.get_table(
        'WIKI/PRICES',
        ticker=ticker,
        date=date
    )
    return df


def save(df, ticker, datadir):
    path = os.path.abspath(os.path.join(datadir, ticker + '.csv.gz'))
    parent = os.path.dirname(path)
    if not os.path.exists(parent):
        os.makedirs(parent)

    df.index = df['date']

    # Date is our new index. So we can drop date and the first column, which is
    # a counter.
    df[df.columns[2:]].to_csv(path, compression='gzip')

# tickers = ['IBM', 'GOOG', 'AAPL', 'TSLA', 'BA', 'AIR', 'MSFT', 'T', 'FDX']
# start_date = '2006-01-01'
# end_date = '2017-05-26'
# for ticker in tickers:
#    data.append(downloader.download(quandl, ticker, start_date, end_date))
#    downloader.save(data[-1], ticker, './data')
