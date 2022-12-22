Problem Definition:

                                     The NIFTY 50 is a benchmark Indian stock market index that represents the weighted average of 50 of the largest Indian companies listed on the National Stock Exchange. So we have the real-time dataset of that 50 companies in NSE ranging from the year when they become a largest companies and till 2021. We have tried to predict the trend of the market after 2019 by using 2021 as the predictable year using ARIMA (Auto Regressive Integrated Moving Average) Model. We have also done a download/print the result as pdf via javascript.

Functional Requirements:     
                                               

•	Data Collection
•	Data Processing
•	Training and Testing 
•	Modeling
•	Predicting
•	Rendering   
•	Download/Print the prediction                                    
                                     
Technical Specification:     
 
•	 Libraries Used for Building Model and Prediction :
	Numpy
	Pandas 
	Matplotlib
	Sklearn, Statsmodels
      
•	 Libraries Used for GUI :
	Flask
	Render_template
	Url_for
	Request
                                     
     
•	Programming Languages Used :
	Python
	HTML
	CSS
	JS

•	Functional Programs:
1.	App.py
2.	Home.html
3.	Result.html

Functional Diagram:

![image](https://user-images.githubusercontent.com/81423983/209104775-2261236a-dd10-4df9-adc8-debb66c8a7d5.png)

System Components/Modules:

1.	Choice From the User: By using flask, the choice of the company to predict is given by the user. This contains several buttons that contained inside cards that represent some name and basic details of the company. Here the dataset path will be sent to the server program app.py.


2.	Importing Libraries: Inside app.py , the path is passed as an argument to the function predict() that will call another function stock() that will predict the stock by using libraries such as pandas, numpy, matplotlib, statsmodels etc.,

3.	Data Processing: The data undergoes pre-processing like cleaning the dataset and removing null and unwanted values.

4.	Test , Train Data: We have separated the test and train data by taking the test data upto the year 2019 and train data from 2021.

5.	Built ARIMA Model: An autoregressive integrated moving average model, a statistical analysis model that used time series data to better understand the dataset or predict future values. The model is built and Graph is displayed.

Output & Visualization:
                                             We have given various plots for each step by step process and we also have an option to print/download the generated result as pdf.
![image](https://user-images.githubusercontent.com/81423983/209104923-30f85182-7412-41fd-a2c8-dc55f22e3fca.png)
![image](https://user-images.githubusercontent.com/81423983/209104974-8e42c90a-d5ca-4803-9722-da3e58fe9554.png)
![image](https://user-images.githubusercontent.com/81423983/209105013-ceb702cf-562b-4286-aa87-adcf543ceebd.png)
![image](https://user-images.githubusercontent.com/81423983/209105038-48312d55-ba09-4f0a-9865-d95d2f5c0f96.png)
![image](https://user-images.githubusercontent.com/81423983/209105073-a968de83-8be4-4595-8fca-a5fce1869927.png)


