import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv("FuelConsumptionCo2.csv");
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']];
msk=np.random.rand(len(df))<0.8;
train=cdf[msk];
test=cdf[~msk];
from sklearn import linear_model;
regr = linear_model.LinearRegression();
train_x=np.asanyarray(train[['ENGINESIZE']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y);
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
from sklearn.metrics import r2_score 
test_x=np.asanyarray(test[['ENGINESIZE']])
test_y=np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat=regr.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='yellow')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

