```Python
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt
import xgboost as xgb
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

```Python
# Create the distance function.
def tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)


# Create Regression Functions
def lw_reg(X, y, xnew, kern, tau, intercept):
    # tau is called bandwidth K((x-x[i])/(2*tau))
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

    #Looping through all X-points
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

# def lowess_reg(x, y, xnew, kern, tau):
#     n = len(x)
#     yest = np.zeros(n)
        
#     w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     
    
#     for i in range(n):
#         weights = w[:, i]
#         b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
#         A = np.array([[np.sum(weights), np.sum(weights * x)],
#                     [np.sum(weights * x), np.sum(weights * x * x)]])
#         theta, res, rnk, s = linalg.lstsq(A, b)
#         yest[i] = theta[0] + theta[1] * x[i] 
#     f = interp1d(x, yest,fill_value='extrapolate')
#     return f(xnew)
```



```Python
# boosted linear weighted regression
def boosted_lwr(X, y, xnew, kern, tau, intercept):
  # we need decision trees
  # for training the boosted method we use X and y
  Fx = lw_reg(X,y,X,kern,tau,intercept) # we need this for training the Decision Tree
  # Now train the Decision Tree on y_i - F(x_i)
  new_y = y - Fx
  tree_model = DecisionTreeRegressor(max_depth=2, random_state=123)
  tree_model.fit(X,new_y)
  output = tree_model.predict(xnew) + lw_reg(X,y,xnew,kern,tau,intercept)
  return output 

# boosted random forest 
def boosted_RF(X, y, xnew):
  model_rf = RandomForestRegressor(n_estimators=100,max_depth=3)
  model_rf.fit(X,y)
  Fx = model_rf.predict(xnew)   
  # Now train the Decision Tree on y_i - F(x_i)
  new_y = y - Fx
  tree_model = DecisionTreeRegressor(max_depth=2, random_state=123)
  tree_model.fit(X,new_y)
  output = tree_model.predict(xnew) + model_rf.predict(xnew)   
  return output 

# neural network
model_nn = Sequential()
model_nn.add(Dense(128, activation="relu", input_dim=3))
model_nn.add(Dense(128, activation="relu"))
model_nn.add(Dense(128, activation="relu"))
model_nn.add(Dense(128, activation="relu"))
model_nn.add(Dense(1, activation="linear"))
model_nn.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1e-2)) # lr=1e-3, decay=1e-3 / 200)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=800)

# boosted Neural Network
def boosted_NN(X, y, xnew):
  model_nn.fit(X,y,validation_split=0.2, epochs=500, batch_size=10, verbose=0, callbacks=[es])
  Fx = model_nn.predict(xnew)
  model_rf.predict(xnew)   
  new_y = y - Fx
  tree_model = DecisionTreeRegressor(max_depth=2, random_state=123)
  tree_model.fit(X,new_y)
  output = tree_model.predict(xnew) + model_nn.predict(xnew)   
  return output 


# boosted Kernal regression
def boosted_KR(data_train, data_test):
  model_KernReg = KernelReg(endog=data_train[:,-1],exog=data_train[:,:-1],var_type='ccc') #,ckertype='gaussian')
  yhat_sm, yhat_std = model_KernReg.fit(data_test[:,:-1])
  Fx = yhat_sm   
  new_y = y - Fx
  tree_model = DecisionTreeRegressor(max_depth=2, random_state=123)
  tree_model.fit(X,new_y)
  output = tree_model.predict(xnew) + yhat_sm 
  return output 
```

```Python
concrete = pd.read_csv('/content/drive/MyDrive/AML/data/concrete.csv')
concrete

# X = concrete.loc[:,'cement':'age']

X = concrete[['cement',	'slag',	'age']].values
# y = concrete['strength'].values

# cars[['ENG','CYL','WGT']].values



y = concrete['strength'].values
```

```Python
# Calculate MSE for Locally Weighted Regression, Boosted Locally Weighted Regression, Random Forest, Extreme Gradient Boosting, Neural Network 

mse_lwr = []
mse_blwr = []
mse_rf = []
mse_brf = []
mse_xgb = []
mse_nn = []
mse_bnn = []
mse_kr = []
mse_bkr = []
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
    yhat_lwr = lw_reg(xtrain,ytrain, xtest,tricubic,tau=0.9,intercept=True)
    yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,tricubic,tau=0.9,intercept=True)
    model_rf = RandomForestRegressor(n_estimators=100,max_depth=3)
    model_rf.fit(xtrain,ytrain)
    yhat_rf = model_rf.predict(xtest)
    # yhat_brf = boosted_RF(xtrain, ytrain, xtest)
    model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
    model_xgb.fit(xtrain,ytrain)
    yhat_xgb = model_xgb.predict(xtest)
    model_nn.fit(xtrain,ytrain,validation_split=0.2, epochs=500, batch_size=10, verbose=0, callbacks=[es])
    yhat_nn = model_nn.predict(xtest)
    # yhat_bnn = boosted_NN(xtrain, ytrain, xtest)
    model_KernReg = KernelReg(endog=data_train[:,-1],exog=data_train[:,:-1],var_type='ccc') #,ckertype='gaussian')
    yhat_sm, yhat_std = model_KernReg.fit(data_test[:,:-1])
    # yhat_bkr = boosted_KR(data_train, data_test)
    # append mse values 
    mse_lwr.append(mse(ytest,yhat_lwr))
    mse_blwr.append(mse(ytest,yhat_blwr))
    mse_rf.append(mse(ytest,yhat_rf))
    # mse_brf.append(mse(ytest, yhat_brf))
    mse_xgb.append(mse(ytest,yhat_xgb))
    mse_nn.append(mse(ytest,yhat_nn))
    # mse_bnn.append(mse(ytest, yhat_bnn))
    mse_kr.append(mse(ytest, yhat_sm))
    # mse_bkr.append(mse(ytest, yhat_bkr))
print('The cross-validated Mean Squared Error for: ')
print('LWR = ' + str(np.mean(mse_lwr)))
print('BLWR = ' + str(np.mean(mse_blwr)))
print('RF = ' + str(np.mean(mse_rf)))
# print('BRF = ' + str(np.mean(mse_brf)))
print('XGB = ' + str(np.mean(mse_xgb)))
print('NN = ' + str(np.mean(mse_nn)))
# print('BNN = ' + str(np.mean(mse_bnn)))
print('KR = ' + str(np.mean(mse_kr)))
# print('BKR = ' + str(np.mean(mse_bkr)))
```

```Python

```

```Python

```
