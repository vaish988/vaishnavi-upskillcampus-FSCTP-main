import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('Traffic data.csv')
df.head()
df
df.isnull().sum()
df.info()
plt.figure(figsize=(10,7))
plt.show()
df.drop(columns=['ID'], axis=1, inplace=True)
df.head()
df['ds'] = df.index
df.head()
size = 60
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=size/len(df), shuffle=False)
train.tail()
test.head()
test.tail()
model = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.9)
model.fit(train)
future = model.make_future_dataframe(periods=60)
future
forecast = model.predict(future)
forecast.head()
model.plot_components(forecast)
pred = forecast.iloc[-60:, :]
len(pred)
plt.figure(figsize=(10,7))
plt.plot(test['ds'], test['y'])
plt.plot(pred['ds'], pred['yhat'], color='red')
plt.plot(pred['ds'], pred['yhat_lower'], color='green')
plt.plot(pred['ds'], pred['yhat_upper'], color='orange')
plt.show()
plt.plot(df['ds'], df['y'])
plt.show()
plt.plot(forecast['ds'], forecast['yhat'])
plt.show()
model = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.9)
model.fit(df)
future = model.make_future_dataframe(periods=200)
forecast = model.predict(future)
forecast.head()
plt.plot(forecast['ds'], forecast['yhat'])
plt.show()
