Stock Price Prediction using KNN
Overview

This project implements a machine learning–based stock price prediction system using historical market data. The objective is to analyze price movements and generate buy/sell signals using classification, along with next-day price estimation using regression.

The project demonstrates end-to-end workflow including data preprocessing, feature engineering, model selection, hyperparameter tuning, and performance evaluation.

Dataset

Source: Historical stock price dataset in Excel format

Records: 1009 trading days

Features:

Open

High

Low

Close

Adjusted Close

Volume

Date

Feature Engineering

The following derived features were created to capture intraday price movement:

Open − Close

High − Low

Date components were also extracted for exploratory analysis:

Year

Month

Day

Problem Formulation
1. Classification Task

Predict whether a stock should be bought (+1) or sold (−1) based on next-day price movement.

Target definition:

Buy if next day closing price is higher than current day

Sell otherwise

Model used:

K-Nearest Neighbors Classifier

Hyperparameter tuning using GridSearchCV

2. Regression Task

Predict the closing price of the stock using historical price patterns.

Model used:

K-Nearest Neighbors Regressor

Hyperparameter optimization with cross-validation

Model Training

Train-test split: 75% training, 25% testing

Scaling applied to training data

Hyperparameters optimized using 5-fold cross-validation

Results
Classification Performance

Training Accuracy: ~75%

Test Accuracy: ~43%

This highlights overfitting challenges and the limitations of distance-based models on noisy financial data.

Regression Performance

Root Mean Squared Error: ~422

Predicted prices capture general trends but struggle with sharp market fluctuations.

Key Learnings

Financial time-series data is highly noisy and non-stationary

KNN models are sensitive to feature scaling and distance metrics

Feature engineering has a significant impact on predictive performance

Simple ML models provide baseline insights but are insufficient for production trading systems

Tech Stack

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Jupyter Notebook

Future Improvements

Use time-series models such as LSTM or ARIMA

Introduce technical indicators like RSI and MACD

Apply walk-forward validation

Improve feature set using rolling statistics

How to Run

Clone the repository

Install dependencies

Run the notebook sequentially

Disclaimer

This project is for educational purposes only and should not be used for real-world trading decisions.
