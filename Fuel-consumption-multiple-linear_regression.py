import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
df = pd.read_csv("FuelConsumptionCo2.csv");
cdf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
msk = np.random.rand(len(df))<0.8
train=cdf[msk]
test=cdf[~msk]
from sklearn import linear_model
regr=linear_model.LinearRegression()
train_x=np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
x=np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y=np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat=regr.predict(x)
print("Residual sum of squares: %.2f"
      % np.mean((test_y_hat - y) ** 2))
print('Variance score: %.2f' % regr.score(x, y))
