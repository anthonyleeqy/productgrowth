# productgrowth
This is a product growth model using variable selection, time series analysis, regressions, principal component analysis (PCA) and long-short term memory (LSTM) model to predict product volume growth.

Main steps are as follows:
1) a data pipeline to ingest data and perform ETL

2) time series test to test the staionality (Time series test.py)

When we get time series data with date, it is always a good idea to test its serial depedencies to have a sense of what model can fit better.

3) a variable selection process and a stepwise regression (Feature selection.py)

Using recursive feature elimination with cross validation to select most relevant features using scikit learn and prevent model overfitting.

4) a linear regressor (OLS.py)

A classic OLS model, usually providing a model benchmark for performance comparison.

5) a PCA process (PCA.py)

Another way to prevent model overfitting.

6) fitting into an LSTM model (LSTM.py)

An RNN model with capability to learn long-term dependencies. Works well especially when we have long datasets from multiple years.

Remember to compare all the model performance and consider all the metrics. A simple model does not mean a bad solution. 
