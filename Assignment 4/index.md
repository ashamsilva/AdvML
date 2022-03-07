# Assignment 4: Part 1 

Import the necessary libraries and initialize StandardScaler() as scale

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

Create the distance function and regression Functions
```Python
def tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)


def lw_reg(X, y, xnew, kern, tau, intercept):
    n = len(X) # the number of observations
    yest = np.zeros(n)

    if len(y.shape)==1: # column vectors
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)
    
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) # here we compute n vectors of weights

    # Run through X-points
    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew) 
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output
```

Create a neural network and a linear weighted regression boosted by Decision Trees

```Python
def boosted_lwr(X, y, xnew, kern, tau, intercept):
  Fx = lw_reg(X,y,X,kern,tau,intercept) # we need this for training the Decision Tree
  new_y = y - Fx
  tree_model = DecisionTreeRegressor(max_depth=2, random_state=123)
  tree_model.fit(X,new_y)
  output = tree_model.predict(xnew) + lw_reg(X,y,xnew,kern,tau,intercept)
  return output 

model_nn = Sequential()
model_nn.add(Dense(128, activation="relu", input_dim=3))
model_nn.add(Dense(128, activation="relu"))
model_nn.add(Dense(128, activation="relu"))
model_nn.add(Dense(128, activation="relu"))
model_nn.add(Dense(1, activation="linear"))
model_nn.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1e-2)) # lr=1e-3, decay=1e-3 / 200)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=800)
```

Read in the data and assign X and y values 
```Python
concrete = pd.read_csv('/content/drive/MyDrive/AML/data/concrete.csv')

X = concrete[['cement',	'slag',	'age']].values
y = concrete['strength'].values
```

Calculate MSE for Locally Weighted Regression, Boosted Locally Weighted Regression, Random Forest, Extreme Gradient Boosting, Neural Network, and Kernal Regression 
```Python
mse_lwr = []
mse_blwr = []
mse_rf = []
mse_xgb = []
mse_nn = []
mse_kr = []
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
    model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
    model_xgb.fit(xtrain,ytrain)
    yhat_xgb = model_xgb.predict(xtest)
    model_nn.fit(xtrain,ytrain,validation_split=0.2, epochs=500, batch_size=10, verbose=0, callbacks=[es])
    yhat_nn = model_nn.predict(xtest)
    model_KernReg = KernelReg(endog=data_train[:,-1],exog=data_train[:,:-1],var_type='ccc') #,ckertype='gaussian')
    yhat_sm, yhat_std = model_KernReg.fit(data_test[:,:-1])
    # append mse values 
    mse_lwr.append(mse(ytest,yhat_lwr))
    mse_blwr.append(mse(ytest,yhat_blwr))
    mse_rf.append(mse(ytest,yhat_rf))
    mse_xgb.append(mse(ytest,yhat_xgb))
    mse_nn.append(mse(ytest,yhat_nn))
    mse_kr.append(mse(ytest, yhat_sm))
print('The cross-validated Mean Squared Error for: ')
print('LWR = ' + str(np.mean(mse_lwr)))
print('BLWR = ' + str(np.mean(mse_blwr)))
print('RF = ' + str(np.mean(mse_rf)))
print('XGB = ' + str(np.mean(mse_xgb)))
print('NN = ' + str(np.mean(mse_nn)))
print('KR = ' + str(np.mean(mse_kr)))
```

### Conclusion
The cross-validated Mean Squared Error for: 
LWR = 81.27281459895187
BLWR = 69.53780060901502
RF = 99.9618424515488
XGB = 51.901217734256235
NN = 82.86838416811096
KR = 58.63711029059199

Kernal Regression was able to produce the lowest MSE is the Extreme Gradient Boosting method with an MSE of 51.90.

# Assignment 4: Part 2

LightGBM optimizes both speed and memory usage. Unlike other boosting tools which use pre-sort-based algorithms, LightGBM uses histogram-based algorithms. This sorts continuous features, or independent variables, into discrete bins. By doing so the process uses less memory storage and takes less time. As mentioned, there are several benefits to histogram-based algorithms which include increased speed. This is because the process uses a fast sum-up operation and once the histogram has been computed, the data is sorted into a set number of bins which is less than the sum of the individual data points. The process of replacing the continuous values of the data with discrete bins is also what drives the reduced memory usage. Another benefit of LightGBM is the use of leaf-wise tree growth over level tree growth. Leaf-wise tree growth splits the best node one level down creating a asymmetric tree. One issue with the level tree growth approach is the possibility of overfitting. This is prevented by setting a max-depth to limit tree depth. LightGBM has also been shown to speed up the
training process of conventional Gradient Boosting Decision Tree.

```Python
# Import additional libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score

data = pd.read_csv('/content/drive/MyDrive/AML/data/concrete.csv')
X = data[['cement',	'slag',	'age']].values
y = data['strength'].values

# scale the data
scale =StandardScaler()
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,random_state=0)
Xtrain = scale.fit_transform(Xtrain)
Xtest = scale.transform(Xtest)

# Convert into LGB Dataset Format
train=lgb.Dataset(Xtrain, label=ytrain)
# Set the parameters 
params={'learning_rate': 0.03, 
        'boosting_type':'gbdt', #GradientBoostingDecisionTree
        'objective':'regression',#regression task
        'n_estimators':100,
        'max_depth':10}
# Create and train the model
clf=lgb.train(params, train,100)
# model prediction 
ypred=clf.predict(Xtest)
# MSE 
mean_squared_error(ypred,ytest)
```

## Conclusion
The MSE found from this method was 52.29









