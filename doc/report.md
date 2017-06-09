% Machine Learning Engineer Nanodegree Capstone Project: Stock Price Predictor
% Renato L. F. Cunha
% June 07th, 2017

# I. Definition
_(approx. 1-2 pages)_

## Project Overview

Statistical models and machine learning have been used in various domains,
ranging from baseball player performance prediction to stock prediction.
In the latter case, investment firms, hedge funds and small investors develop
or follow financial models to understand and, to some extend, predict market
behavior to make profitable investments.

In this project we will exploit the wealth of historical stock data available
on the Internet to solving the problem of predicting stock prices. In doing so,
we will investigate the performance of various models for predicting stock
prices for publicly-traded American companies. The data we want to predict
looks like the one presented in Figure \ref{ibm-stock}, which shows the
close price[^1] for IBM stock for the last 10 years.

[^1]: The close price is the price of the stock at the end of a business day.
  As will be made clear later, we do not use the exact close price, but
  a metric derived from it.

![IBM stock prices for the last 10 years.\label{ibm-stock}](img/ibm-stock.png)

More specifically, the data consists of tables containing, for each stock, for
each day: the opening price, the highest price, the lowest price, the close
price and the number of transactions. The objective is to use this data to make
predictions for stock prices $N$ days in advance, with $N$ in the range $[1, 5,
10, 21]$. An excerpt of the data is shown below[^2].

|date        | open   | high   | low     | close |      volume |
|------------|--------|--------|---------|-------|-------------|
|2006-01-03  | 82.45  | 82.55  | 80.810  | 82.06 |  11715100.0 |
|2006-01-04  | 82.20  | 82.50  | 81.330  | 81.95 |   9832800.0 |
|2006-01-05  | 81.40  | 82.90  | 80.999  | 82.50 |   7213400.0 |
|2006-01-06  | 83.95  | 85.03  | 83.410  | 84.95 |   8196900.0 |
|2006-01-09  | 83.90  | 84.25  | 83.380  | 83.73 |   6851100.0 |

[^2]: Notice there is a jump from 2006-01-06 to 2006-01-09. That is due to the
  fact we only record data for weekdays, and not weekends.

## Problem Statement

The problem tackled by this project is that of predicting stock prices for
future dates given historical data about such stock items. Inputs will contain
multiple metrics, such as opening price (Open), highest price the stock traded
at (High), how many stocks were traded (Volume) and closing price adjusted for
stock splits and dividends (Adjusted Close). The objective in this project is
to predict the Adjusted Close price. The simplest solution would be to predict
the mean value of the adjusted close price, but clearly we can strive to do
better than that.

One might wonder why predict for Adjusted Close instead of just the close
price. There are at least two reasons to use an adjusted value instead of the
raw one:

 1. When individual stocks become too expensive, the company may want to split
    stocks, reducing the individual price of a single stock. When that happens,
    the price will drop, making prediction harder. For example, if an algorithm
    was trained when a given stock cost around $ 100, but then there is a split
    of 2, now each stock item costs $ 50, and the previous training data
    becomes "useless". More information on this topic can be found [at
    Wikipedia](https://en.wikipedia.org/wiki/Stock_split).

 2. Some stocks pay [dividends](https://en.wikipedia.org/wiki/Dividend) at
    previously-determined dates. Due to that, demand for such stock increases
    as the dividend payment date approaches, artificially inflating the stock
    price. After the dividends are paid, the stock price converges once again
    to its actual price.

Due to the aforementioned reasons, we work with adjusted stock prices, by
working backwards in time updating prices considering splits and dividends. One
consequence of applying such a method, though, is that the adjusted prices of
*all* stock items will change when a dividend or split is found. Therefore, when
new data arrives, models may have to be retrained. Also, models should output
predicted adjusted close prices in dollars (a real number).

All the major providers (examples include [Yahoo!
Finance](https://finance.yahoo.com) and [Quandl](https://www.quandl.com/)) of
historical stock data already provide adjusted stock prices. Hence, no
computation is needed on our part to compute adjusted values.

### Feature Engineering

Although we don't *have* to compute adjusted prices, we can augment our data by
computing useful statistics. In this project we will focus on the following
ones:

 1. The *rolling mean* gives us the average value of a stock in the last $n$
    days (in this project we will use $n=21$, roughly a month in business
    days);
 2. Bollinger bands and metrics derived from it, such as
    [*bandwidth*](https://en.wikipedia.org/wiki/Bollinger_Bands#Indicators_derived_from_Bollinger_Bands)
    and %b, in the hope of identifying opportunities in the valuation of
    a stock;
 3. [*Momentum*](https://en.wikipedia.org/wiki/Momentum_(finance)), which
    indicates the trend of a given stock;
 4. [*Volatility*](https://en.wikipedia.org/wiki/Volatility_(finance)), which
    represents the degree of variation of a trading price over time;
 5. [*Beta*](https://en.wikipedia.org/wiki/Beta_(finance)), which indicates
    whether a stock is more or less volatile than the market as a whole[^3].

[^3]: Since we do not have data about the whole market, in this project we will
  use the S&P 500 prices as a proxy for the performance of the market as
  a whole.

As can be seen from the references, all of the aforementioned features are used
in finance, and seem to be relevant to the problem.

## Metrics

Since this is a regression problem, for we are predicting a single number, we
should use a metric that works correctly with regressions. Initially, it was
thought that mean squared error would be a good metric, but it only allows the
ordering of the quality of models within one dataset. In short: the mean
squared error (and even the root mean squared error) is larger or shorter
depending on the magnitude of values.

For the reasons above, we will be using the [*coefficient of
determination*](https://en.wikipedia.org/wiki/Coefficient_of_determination), or
$R^2$ as the performance measurement metric. The $R^2$ score has the advantage
of being independent of magnitude of the data and of being standardize, where
1 is the score of a model that perfectly fits the data.

# II. Analysis
_(approx. 2-4 pages)_

## Data Exploration

For obtaining the data we used the [Quandl
API](https://pypi.python.org/pypi/Quandl) and we downloaded the data for the
last ten years for the following tickers: 'IBM' (IBM), 'GOOG' (Google), 'AAPL'
(Apple), 'TSLA' (Tesla), 'BA' (Boeing), 'MSFT' (Microsoft), 'T' (AT&T), 'AIR'
(AAR Corp.) and 'FDX' (Fedex). For obtaining the data about S&P 500 we used the
Yahoo! Finance interface for download, since this data is not available in the
free plan of Quandl.

As already mentioned, the data is tabular data, has one entry for weekday and 

In this section, you will be expected to analyze the data you are using for the
problem. This data can either be in the form of a dataset (or datasets), input
data (or input files), or even an environment. The type of data should be
thoroughly described and, if possible, have basic statistics and information
presented (such as discussion of input features or defining characteristics
about the input or environment). Any abnormalities or interesting qualities
about the data that may need to be addressed have been identified (such as
features that need to be transformed or the possibility of outliers). Questions
to ask yourself when writing this section:

- _If a dataset is present for this problem, have you thoroughly discussed
  certain features about the dataset? Has a data sample been provided to the
  reader?_
- _If a dataset is present for this problem, are statistics about the dataset
  calculated and reported? Have any relevant results from this calculation been
  discussed?_
- _If a dataset is **not** present for this problem, has discussion been made
  about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or
  dataset that need to be addressed? (categorical variables, missing values,
  outliers, etc.)_

## Exploratory Visualization

To better appreciate the difference between the adjusted and non-adjusted
prices of stock, one is directed to Figure \ref{apple-stock}. In the Figure, on
the left, one can see that at some point in 2014 there was a sharp price drop
in the Apple stock price due to share splits. This sharp change does not happen
in the image on the right, due to the split being taken into account when
computing prices.

![Apple stock prices for the last 10 years.\label{apple-stock}](img/apple-stock.png)

![Trends in stock prices. A sharp drop in 2008 (recession) and one seemingly
seasonal stock.\label{four-stocks}](img/stocks.png)

In this section, you will need to provide some form of visualization that
summarizes or extracts a relevant characteristic or feature about the data. The
visualization should adequately support the data being used. Discuss why this
visualization was chosen and how it is relevant. Questions to ask yourself when
writing this section:

- _Have you visualized a relevant characteristic or feature about the dataset
  or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

## Algorithms and Techniques

In this section, you will need to discuss the algorithms and techniques you
intend to use for solving the problem. You should justify the use of each one
based on the characteristics of the problem and the problem domain. Questions
to ask yourself when writing this section:

- _Are the algorithms you will use, including any default variables/parameters
  in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the
  algorithms and techniques chosen?_

## Benchmark

In this section, you will need to provide a clearly defined benchmark result or
threshold for comparing across performances obtained by your solution. The
reasoning behind the benchmark (in the case where it is not an established
result) should be discussed. Questions to ask yourself when writing this
section:

- _Has some result or value been provided that acts as a benchmark for
  measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by
  hypothesis)?_


# III. Methodology
_(approx. 3-5 pages)_

## Data Preprocessing

In this section, all of your preprocessing steps will need to be clearly
documented, if any were necessary. From the previous section, any of the
abnormalities or characteristics that you identified about the dataset will be
addressed and corrected here. Questions to ask yourself when writing this
section:

- _If the algorithms chosen require preprocessing steps like feature selection
  or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or
  characteristics that needed to be addressed, have they been properly
  corrected?_
- _If no preprocessing is needed, has it been made clear why?_

## Implementation

In this section, the process for which metrics, algorithms, and techniques that
you implemented for the given data will need to be clearly documented. It
should be abundantly clear how the implementation was carried out, and
discussion should be made regarding any complications that occurred during this
process. Questions to ask yourself when writing this section:

- _Is it made clear how the algorithms and techniques were implemented with the
  given datasets or input data?_
- _Were there any complications with the original metrics or techniques that
  required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated
  functions) that should be documented?_

## Refinement

In this section, you will need to discuss the process of improvement you made
upon the algorithms and techniques you used in your implementation. For
example, adjusting parameters for certain models to acquire improved solutions
would fall under the refinement category. Your initial and final solutions
should be reported, as well as any significant intermediate results as
necessary. Questions to ask yourself when writing this section:

- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques
  were used?_
- _Are intermediate and final solutions clearly reported as the process is
  improved?_

# IV. Results
_(approx. 2-3 pages)_

## Model Evaluation and Validation

In this section, the final model and any supporting qualities should be
evaluated in detail. It should be clear how the final model was derived and why
this model was chosen. In addition, some type of analysis should be used to
validate the robustness of this model and its solution, such as manipulating
the input data or environment to see how the model’s solution is affected (this
is called sensitivity analysis). Questions to ask yourself when writing this
section:

- _Is the final model reasonable and aligning with solution expectations? Are
  the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the
  model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes)
  in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

## Justification

In this section, your model’s final solution and its results should be compared
to the benchmark you established earlier in the project using some type of
statistical analysis. You should also justify whether these results and the
solution are significant enough to have solved the problem posed in the
project. Questions to ask yourself when writing this section:

- _Are the final results found stronger than the benchmark result reported
  earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


# V. Conclusion
_(approx. 1-2 pages)_

## Free-Form Visualization

In this section, you will need to provide some form of visualization that
emphasizes an important quality about the project. It is much more free-form,
but should reasonably support a significant result or characteristic about the
problem that you want to discuss. Questions to ask yourself when writing this
section:

- _Have you visualized a relevant or important quality about the problem,
  dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

## Reflection

In this section, you will summarize the entire end-to-end problem solution and
discuss one or two particular aspects of the project you found interesting or
difficult. You are expected to reflect on the project as a whole to show that
you have a firm understanding of the entire process employed in your work.
Questions to ask yourself when writing this section:

- _Have you thoroughly summarized the entire process you used for this
  project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and
  should it be used in a general setting to solve these types of problems?_

## Improvement

In this section, you will need to provide discussion as to how one aspect of
the implementation you designed could be improved. As an example, consider ways
your implementation can be made more general, and what would need to be
modified. You do not need to make this improvement, but the potential solutions
resulting from these changes are considered and compared/contrasted to your
current solution. Questions to ask yourself when writing this section:

- _Are there further improvements that could be made on the algorithms or
  techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how
  to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even
  better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure
  similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in
  a clear, concise and specific fashion? Are there any ambiguous terms or
  phrases that need clarification?
- Would the intended audience of your project be able to understand your
  analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal
  grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly
  commented?
- Does the code execute without error and produce results similar to those
  reported?
