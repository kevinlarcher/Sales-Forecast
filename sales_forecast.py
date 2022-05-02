import fbprophet as fp
import pandas as pd
#from matplotlib import pyplot
#import seaborn as sns
#from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv('Sales_Data.csv', header=0)
data.columns = ['ds','y']
data.set_index('ds')
data.index
data.index = pd.to_datetime(data.index)
data = data.reset_index(drop=True)

y = data

model = fp.Prophet(weekly_seasonality=True, daily_seasonality=True)
model.fit(y)

y_f = model.make_future_dataframe(periods=13, freq='m')

pred = model.predict(y_f)
pred.to_csv('Results.csv')

forecast = pred.tail(12)
forecast = forecast[['ds','yhat']]
forecast.columns = ['ds','sales forecast']
forecast.to_csv('Forecast.csv')
