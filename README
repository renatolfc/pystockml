# pystockml

This is pystockml a python project that does stock price predictions using
machine learning.

## Requirements

If you are on a Python VirtualEnv or have a Python installation where you can
write to the system, you can install the project's requirements with:

```
pip install -r requirements.txt
```

When you do so, pip will install:

 * Quandl for downloading stock data
 * statsmodels for implementing an ARIMA model
 * Keras for implementing MLP and LSTM neural networks
 * pandas for loading and manipulating data
 * scikit-learn for building the machine learning models
 * seaborn for pretty graphs

Notice that the list above is not exhaustive, but pip is able to fetch
dependencies, so this should be enough for you.

## Code structure

### Client program

Most of the functionality is implemented in the `pystockml` package, and
a client program is available at `predict.py`. `predict.py` is a python
command-line application that is able to train models and predict stock prices.
You can read `predict.py`'s help by calling it as `./predict.py -h`.  To
predict Tesla stock prices 33 days after 2016-09-08 using training data
beginning at Tesla's IPO (which happened after 2010-01-01), one would do:

```
$ ./predict.py TSLA 2010-01-01 2016-09-08 33
Using TensorFlow backend.
Loading data...
Training model...
Predicting...
Predicted value: 211.276317094
```

### pystockml

Pystockml is, at is core, comprised of three modules:

 * `downloader.py`: A module that uses the Quandl API to download stock data;
 * `statistics.py`: A module that implements functions that augment the data
                    downloaded from Quandl by computing finance-related
                    statistics.
 * `models.py`: A model that implements machine learning models for stock price
                prediction alongside functions for preprocessing the data.
