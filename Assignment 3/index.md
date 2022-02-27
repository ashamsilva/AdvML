# Multivariate Regression Analysis and Gradient Boosting

### Multivariate Regression Analysis

Multivariate regression analysis is a more robust continuation of a linear regression analysis and has more than one independent variable. Unlike linear regressions, we do not have to make assumptions regarding predictors. Linear regressions which define y as a function of x. Multivariate regression analysis allows us to define y as a function of multiple variables (ie. x and z) such that y = f(x,z). In adding this additional variable, an additional dimension is added as well. In the example that y is a function of x and z a plane would be predicted. 

The equation for the predictions we make is:
y = B0 + B1*X1 + B2*X2 + ... + Bn+Xn

### Gradient Boosting

Gradient boosting, or gbm, is a method used in machine learning to increase the stregth of learners. When gradient boosting is used, each additional tree is fit on a version of the orginal dataset that has been modified. Using these decision trees, the model is trained and produces a final prediction value of the dependent variable. This process takes the mean from the original dataset and adds the residuals predicted by the decision trees in the forest. Extreme Gradient Boosting is a derivative of gradient boosting that uses regularization parameters in order to prevent overfitting.

## Application of Locally Weighted Regression and Random Forest on Datasets

### Funtions and Code used for Both Datasets 

Import the necessary libraries and assign StandardScaler() as scale.
```Python
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from statsmodels.nonparametric.kernel_regression import KernelReg
import matplotlib.pyplot as plt
import lightgbm as lgb
from matplotlib import pyplot
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import r2_score
from tensorflow.keras.optimizers import Adam, SGD, RMSprop # they recently updated Tensorflow
from keras.callbacks import EarlyStopping

scale = StandardScaler()
```

Create the distance function.
```Python
def tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)
```

Create Regression Functions
```Python
def lw_reg(X, y, xnew, kern, tau, intercept):
    n = len(X) # the number of observations
    yest = np.zeros(n)

    if len(y.shape)==1: # here we make column vectors
      y = y.reshape(-1,1)
    if len(X.shape)==1:
      X = X.reshape(-1,1)
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) # here we compute n vectors of weights

for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        #A = A + 0.001*np.eye(X1.shape[1]) # if we want L2 regularization
        #theta = linalg.solve(A, b) # A*theta = b
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew) # the output may have NaN's where the data points from xnew are outside the convex hull of X
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      # output[np.isnan(output)] = g(X[np.isnan(output)])
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output


def lowess_reg(x, y, xnew, kern, tau):
    n = len(x)
    yest = np.zeros(n)
        
    w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     
    
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        theta, res, rnk, s = linalg.lstsq(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 
    f = interp1d(x, yest,fill_value='extrapolate')
    return f(xnew)


def boosted_lwr(X, y, xnew, kern, tau, intercept):
  # we need decision trees
  # for training the boosted method we use X and y
  Fx = lw_reg(X,y,X,kern,tau,intercept) 
  # Now train the Decision Tree on y_i - F(x_i)
  new_y = y - Fx
  model = RandomForestRegressor(n_estimators=100,max_depth=2)
  model.fit(X,new_y)
  output = model.predict(xnew) + lw_reg(X,y,xnew,kern,tau,intercept)
  return output 
 ```

Calculate MSE for Locally Weighted Regression, Boosted Locally Weighted Regression, Random Forest, Extreme Gradient Boosting, Neural Network and Nadarya-Watson Regressor
```Python
mse_lwr = []
mse_blwr = []
mse_rf = []
mse_xgb = []
mse_nn = []
mse_NW = []
for i in [10]:
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
    data_train = np.concatenate([xtrain,ytrain.reshape(-1,1)],axis=1)
    data_test = np.concatenate([xtest,ytest.reshape(-1,1)],axis=1)
    yhat_lwr = lw_reg(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
    yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
    model_rf = RandomForestRegressor(n_estimators=100,max_depth=3)
    model_rf.fit(xtrain,ytrain)
    yhat_rf = model_rf.predict(xtest)
    model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
    model_xgb.fit(xtrain,ytrain)
    yhat_xgb = model_xgb.predict(xtest)
    model_nn.fit(xtrain,ytrain,validation_split=0.2, epochs=500, batch_size=10, verbose=0, callbacks=[es])
    yhat_nn = model_nn.predict(xtest)
    model_KernReg = KernelReg(endog=data_train[:,-1],exog=data_train[:,:-1],var_type='ccc',ckertype='gaussian')
    yhat_sm, yhat_std = model_KernReg.fit(data_test[:,:-1])
    mse_lwr.append(mse(ytest,yhat_lwr))
    mse_blwr.append(mse(ytest,yhat_blwr))
    mse_rf.append(mse(ytest,yhat_rf))
    mse_xgb.append(mse(ytest,yhat_xgb))
    mse_nn.append(mse(ytest,yhat_nn))
    mse_NW.append(mse(ytest,yhat_sm))
Print('The cross-validated Mean Squared Error for: ')
print('LWR = ' + str(np.mean(mse_lwr)))
print('BLWR = ' + str(np.mean(mse_blwr)))
print('RF = ' + str(np.mean(mse_rf)))
print('XGB = ' + str(np.mean(mse_xgb)))
print('NN = ' + str(np.mean(mse_nn)))
print('Nadarya-Watson Regressor = ' + str(np.mean(mse_NW)))
```

### Dataset 1: Housing

Import the data and assign X and y
```Python
housing = pd.read_csv('/content/drive/MyDrive/AML/data/Boston Housing Prices.csv')
X = housing[['longitude','latitude','nox']].values
y = housing['cmedv'].values

data = np.concatenate([X,y.reshape(-1,1)],axis=1)
```

#### Conclusion:

The minimum Cross-validated Mean Squared Error is 41.89 and was found using Extreme Gradient Boosting.

The cross-validated Mean Squared Error for:
LWR = 54.90110127773951
RF is : 51.33316103879288
XGB is : 41.89036168258404
NN is : 44.70710846876278
Nadarya-Watson Regressor = 45.506043382147325


### Dataset 2: Concrete

Import the data and assign X and y
```Python
# import the data
concrete = pd.read_csv('/content/drive/MyDrive/AML/data/concrete.csv')
X = concrete[['cement',	'slag',	'age']].values
y = concrete['strength'].values

data = np.concatenate([X,y.reshape(-1,1)],axis=1)
```


#### Conclusion:

The minimum Cross-validated Mean Squared Error is 51.90 and was found using Extreme Gradient Boosting.

The cross-validated Mean Squared Error for 
LWR = 85.89122686024714
BLWR = 70.1110505540548
RF = 100.35401593594011
XGB = 51.901217734256235
NN = 92.02494297501055
Nadarya-Watson Regressor = 58.63711029059199
