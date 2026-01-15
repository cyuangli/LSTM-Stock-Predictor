
# LSTM Stock Predictor

## Overview

A pipeline designed to customize an LSTM model on predicting market data for a given stock option. 

## Project Steps

1. Obtaining OHLC data from Yahoo Finance

2. Performing EDA and Data Transformations

    2.1 Using PCA biplots, dendrograms, linear correlation analysis, mutual information regression, and heatmaps to visualize and select features with high predictive potential.

3. Model training

	3.1 Using an LSTM model to train on engineered features
    3.2 Using statistical evaluation metrics like the coefficient of determination, MSE, etc, but also market metrics like the profit factor and Sharpe ratio.
    3.3 Using a Monte Carlo permutation test to further test data.

4. Constrution of training and prediction pipeline


## Project Process

### Engineering Features

Feature engineering mostly took place within the notebooks folder. A set of metrics were generated using various technical indicators like volatility, momentum, and volume. By building all of the indicators, a complete analysis performed on them could find important correlations between features and the stock returns. However, surprisingly, it seems like the correlation between most features and the returns are not constant throughout the different visualization tests, and the best model trained was dependent on the OHLC data. 

The explanation seems to be that these manufactured features introduced more noise within the LSTM model. Raw historical data proved to be much better at aiding the LSTM model at predicting future returns, and adding more features simply drowned out these much more important ones. Many features were also redundant, either measuring similar trends as others or providing similar levels of statisical insight as the original OHLC features.




