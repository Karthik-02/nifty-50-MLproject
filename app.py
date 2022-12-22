from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


app=Flask(__name__,template_folder='Templates')
picFolder = os.path.join('stock_prj','pics')
app.config['UPLOAD_FOLDER'] = picFolder

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
	

	if request.method == 'POST':
		message = request.form['message']
		data = message
		my_prediction = 1
		my_head, my_train, my_test , res, mbuild = stock(data)
		stock_img = os.path.join(app.config['UPLOAD_FOLDER'], 'StockPrice.jpg')

	return render_template('result.html', data = data, prediction = my_prediction, head = my_head, train = my_train, test = my_test, adf = res, modbuild = mbuild)


def stock(data):
	df = pd.read_csv(data)
	df.replace(np.nan, inplace=True)
	head = df.head(5)
	head = head.values.tolist()
	df = df[['Date','Close']]
	df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
	plt.figure(figsize=(14,7))
	plt.title('Stock Prices')
	plt.xlabel('Dates')
	plt.ylabel('Close Prices')
	plt.plot(df['Close'])
	plt.xticks(np.arange(0,150, 300), df['Date'][0:150:300])
	plt.savefig("E:/ML/stock_prj/static/StockPrice.jpg")
	train_data, test_data = df[0:int(len(df)*0.9)], df[int(len(df)*0.94):]
	plt.figure(figsize=(14,7))
	plt.title('Stock Prices')
	plt.xlabel('Dates')
	plt.ylabel('Close Prices')
	plt.plot(df['Close'], 'blue', label='Training Data')
	plt.plot(test_data['Close'], 'green', label='Testing Data')
	plt.xticks(np.arange(0,1500, 300), df['Date'][0:1500:300])
	plt.legend()
	plt.savefig("E:/ML/stock_prj/static/TrainTest.jpg")
	tdata = train_data.tail().values.tolist()
	tedata = test_data.head().values.tolist()
	# Check if price series is stationary
	from statsmodels.tsa.stattools import adfuller
	result = adfuller(df.Close.dropna())
	from pmdarima.arima.utils import ndiffs
	ndiffs(df.Close, test="adf")
	from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
	diff = df.Close.diff().dropna()
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
	ax1.plot(diff)
	ax1.set_title("Difference once")
	ax2.set_ylim(0, 1)
	plot_pacf(diff, ax=ax2);
	plt.savefig("E:/ML/stock_prj/static/P.jpg")
	diff = df.Close.diff().dropna()
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
	ax1.plot(diff)
	ax1.set_title("Difference once")
	ax2.set_ylim(0, 1)
	plot_acf(diff, ax=ax2);
	plt.savefig("E:/ML/stock_prj/static/Q.jpg")
	dataset=df.copy()
	dataset.set_index('Date', inplace=True)
	dataset = dataset[['Close']] 
	from matplotlib import pyplot
	pyplot.figure()
	pyplot.subplot(211)
	plot_acf(dataset, ax=pyplot.gca(),lags=10)
	pyplot.subplot(212)
	plot_pacf(dataset, ax=pyplot.gca(),lags=10)
	pyplot.savefig("E:/ML/stock_prj/static/Auto-PartialCore.jpg")
	import statsmodels.api as smapi
	train_ar = train_data['Close'].values
	test_ar = test_data['Close'].values
	history = [x for x in train_ar]
	print(type(history))
	predictions = list()
	for t in range(len(test_ar)): 
		model = smapi.tsa.arima.ARIMA(history, order=(2,1,1))
		model_fit = model.fit()
		output = model_fit.forecast()
		yhat = output[0]
		predictions.append(yhat)
		obs = test_ar[t]
		history.append(obs)
	error = mean_squared_error(test_ar, predictions)
	print('Testing Mean Squared Error: %.3f' % error)
	error2 = smape_kun(test_ar, predictions)
	print('Symmetric mean absolute percentage error: %.3f' % error2)
	build = model_fit.summary()
	plt.figure(figsize=(14,7))
	plt.plot(df['Close'], 'green', color='blue', label='Training Data')
	plt.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed', label='Predicted Price')
	plt.plot(test_data.index, test_data['Close'], color='red', label='Actual Price')
	plt.title(' Prices Prediction')
	plt.xlabel('Dates')
	plt.ylabel('Prices')
	plt.xticks(np.arange(0,1500, 300), df['Date'][0:1500:300])
	plt.legend()
	plt.savefig("E:/ML/stock_prj/static/Prediction.jpg")
	return head, tdata, tedata , result, build

def smape_kun(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))))

if __name__ == '__main__':
	app.run(debug=True)