# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 12:32:14 2019

@author: ms186162
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#read csv
ts = pd.read_csv('weatherseatle.csv', parse_dates=True, index_col = 'Date')
ts.describe()
ts.head()
ts.isna().sum()
ts.index.min(), ts.index.max()
ts = ts.filter(['TotalSeaportsWeight'])
ts.plot()
#ts.groupby(lambda x:x.year)['TotalSeaportsCIF'].agg(['sum','mean','max'])
#https://stackoverflow.com/questions/47399830/show-plots-in-new-window-instead-of-inline-not-answered-by-previous-posts

#from statsmodels.tsa.seasonal import seasonal_decompose
##https://stackoverflow.com/questions/47609537/seasonal-decompose-in-python
##Additive Model = Level + Trend + Seasonality + Noise
#result = seasonal_decompose(ts,model='additive', freq=12)
#result.plot()
#
#result = seasonal_decompose(ts,model='multiplicate', freq=12)
#result.plot()
#
##plot mean and standard deviation for the trend
##Determing rolling statistics
#ts.rolling(12).mean().plot()
#ts.rolling(12).var().plot()
#
#resultplot = pd.concat([ts,ts.rolling(12).mean(),ts.rolling(12).std()], axis=1)
#resultplot.plot()
#
#ts.plot(figsize=(20,10), linewidth=5, fontsize=20)

# dftest = adfuller(timeseries, autolag='AIC')


#Approach 1: removing trend by differencing
ts.dropna(inplace=True)
ts_log = np.log(ts)
ts_log.plot()
tsdiff = ts_log - ts_log.shift() #order 1
tsdiff.dropna(inplace=True)
tsdiff.plot()


from statsmodels.tsa.seasonal import seasonal_decompose
##https://stackoverflow.com/questions/47609537/seasonal-decompose-in-python
##Additive Model = Level + Trend + Seasonality + Noise
result = seasonal_decompose(ts_log,model='additive', freq=4)
result.plot()

tsseasonl = ts_log- result.seasonal-result.trend
original = tsdiff+tsseasonl
original.plot()
result = seasonal_decompose(original.dropna(),model='additive', freq=4)
result.plot()

#lets transform the data into log

#tsdiff.plot(figsize=(20,10), linewidth=5, fontsize=20)
#plt.xlabel('Year', fontsize=20);
#plt.show()
#
## Approach 2: removing trend by linear regression
#from sklearn.linear_model import LinearRegression
#
#
## fit linear model
#X = [i for i in range(0, len(ts))]
#X = np.reshape(X, (len(X), 1))
#y = ts_log.values
#model = LinearRegression()
#model.fit(X, y)
## calculate trend
#trend = model.predict(X)
## plot trend
#plt.plot(y)
#plt.plot(trend)
#plt.show()
## detrend
#detrended = y-trend
## plot detrended
#plt.plot(detrended)
#plt.show()
#
##checking stationarity by statistical tests
#import statsmodels.tsa.stattools as tsastat
#
#dftest = tsastat.adfuller(ts['TotalSeaportsCIF'], 1) # maxlag is  set to 1
#dfoutput = pd.Series(dftest[0:4], index=['ADF Statistic', 'p-value', '#Lags Used', 'Number of Obs Used'])
#for key, value in dftest[4].items():
#    dfoutput['Critical Value (%s)' % key] = value
#    print(dfoutput)
#    
##adfuller test for detrended data
#dftest = tsastat.adfuller(tsdiff['TotalSeaportsCIF'].dropna(), 1) # maxlag is  set to 1
#dfoutput = pd.Series(dftest[0:4], index=['ADF Statistic', 'p-value', '#Lags Used', 'Number of Obs Used'])
#for key, value in dftest[4].items():
#    dfoutput['Critical Value (%s)' % key] = value
#    print(dfoutput)
#
##results show test static > critical value for trending series
##next step in fitting an ARIMA model is to determine whether AR or MA terms are needed to correct any autocorrelation that remains in the differenced series.
##to find the p,q values we need to draw ACF and PACF plots
#from statsmodels.tsa.stattools import acf, pacf
#acfresult = acf(tsdiff['TotalSeaportsCIF'].dropna(), nlags=10)
#pacfresult = pacf(tsdiff['TotalSeaportsCIF'].dropna(), nlags=10, method='ols')
#
##Plot ACF: 
#plt.subplot(121) 
#plt.plot(acfresult)
#plt.axhline(y=0,linestyle='--',color='gray')
#plt.axhline(y=-1.96/np.sqrt(len(tsdiff)),linestyle='--',color='gray')
#plt.axhline(y=1.96/np.sqrt(len(tsdiff)),linestyle='--',color='gray')
#plt.title('Autocorrelation Function')
#
##Plot PACF:
#plt.subplot(122)
#plt.plot(pacfresult)
#plt.axhline(y=0,linestyle='--',color='gray')
#plt.axhline(y=-1.96/np.sqrt(len(tsdiff)),linestyle='--',color='gray')
#plt.axhline(y=1.96/np.sqrt(len(tsdiff)),linestyle='--',color='gray')
#plt.title('Partial Autocorrelation Function')
#plt.tight_layout()

#https://stats.stackexchange.com/questions/84076/negative-values-for-aic-in-general-mixed-model
original.dropna(inplace=True)
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log, order=(1, 1,1))  
results_ARIMA = model.fit(disp=-1)
plt.plot(original)  
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-original['TotalSeaportsWeight'].dropna())**2))
plt.show()
#
#model = ARIMA(tsdiff['TotalSeaportsCIF'].dropna(inplace=True), order=(2, 0, 1))  
#results_ARIMA = model.fit(disp=-1)  
#plt.plot(tsdiff['TotalSeaportsCIF'])
#plt.plot(results_ARIMA.fittedvalues, color='red')
#plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-tsdiff['TotalSeaportsCIF'].dropna())**2))
#results_ARIMA.aic
#we got fitted values from : log and then diff of order 1, so we need to reverse them back
results_ARIMA.fittedvalues.head()
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff.head()
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.head()
ts_log.head()
ts_log.index
predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
predictions_ARIMA_log.head()
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.dropna(inplace=True)

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts_log)
plt.plot(predictions_ARIMA_log)
    




