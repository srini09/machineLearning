# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 11:17:10 2019

@author: ms186162
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#read csv
data = pd.read_csv('weatherseatle.csv')
data.columns
data['Date'] = pd.to_datetime(data['Date'], format="%Y/%m")
data = data.sort_values(by = 'Date')
data = data.set_index('Date')
ts = data['TotalAirportsWeight']
ts.describe()
ts.head()
ts_log = np.log(ts)
ts_log.plot()


monthlydata =ts_log.groupby(ts_log.index.month).mean()
#ts_log = monthlydata
#monthlydata.plot()

from statsmodels.tsa.seasonal import seasonal_decompose
##https://stackoverflow.com/questions/47609537/seasonal-decompose-in-python
##Additive Model = Level + Trend + Seasonality + Noise
result = seasonal_decompose(ts_log,model='additive', freq=4)
result.plot()



ts_log_dif = ts_log.diff(4)
ts_log_dif.dropna(inplace=True)
ts_log_dif.plot()

#ACF and PACF plots:
#https://newonlinecourses.science.psu.edu/stat510/node/62/
#Identification of an AR model is often best done with the PACF.
#Identification of an MA model is often best done with the ACF rather than the PACF
from statsmodels.tsa.stattools import acf, pacf  

lag_acf = acf(ts_log, nlags=6)
lag_pacf = pacf(ts_log, nlags=6, method='ols')
plt.subplot(121)    
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log_dif,order=(4,0,4))
result_MA = model.fit()
plt.plot(ts_log_dif)
plt.plot(result_MA.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((result_MA.fittedvalues-ts_log_dif)**2))
plt.show()


pred_arima_dif = pd.Series(result_MA.fittedvalues,copy=True)
arima_dif_cumsum = pred_arima_dif.cumsum()

pred_arima_log = pd.Series(ts_log.ix[0], index=ts_log.index)
pred_arima_log = pred_arima_log.add(arima_dif_cumsum,fill_value=0)
print(pred_arima_log.head())

pred = np.exp(pred_arima_log)
plt.plot(ts)
plt.plot(pred)
plt.show()