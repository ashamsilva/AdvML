# Comparison of Locally Weighted Regression and Random Forest


Create a new Github page with a presentation on the concepts of Locally Weighted Regression and Random Forest. 
Apply the regression methods to real data sets, such as "Cars" or "Boston Housing Data" where you consider only one input variable (the weight of the car for the "Cars" data set and the number of rooms for the "Boston Hausing" data). The output varable is the mileage for the "Cars" data set and "cmedv" or the median price for the housing data.
For each method and data set report the crossvalidated mean square error and determine which method is achieveng the better results.
In this paper you should also include theoretical considerations, examples of Python coding and plots. 
The final results should be clearly stated.


## Locally Weighted Regression 

## Random Forest 



# Application of Locally Weighted Regression and Random Forest on Datasets

## Funtions and Code used for Both Datasets 

Import the necessary libraries and assign StandardScaler() as scale 
```Python
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
```


```Python
def tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)
```

```Python
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
```


```Python
mse_lwr = []
mse_rf = []
rf = RandomForestRegressor(n_estimators=150,max_depth=3)
kf = KFold(n_splits=10,shuffle=True,random_state=1234)
for idxtrain,idxtest in kf.split(x):
  ytrain = y[idxtrain]
  xtrain = x[idxtrain]
  xtrain = scale.fit_transform(xtrain.reshape(-1,1))
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtest = scale.transform(xtest.reshape(-1,1))
  yhat_lwr = lowess_reg(xtrain.ravel(),ytrain,xtest.ravel(),tricubic,0.4)
  rf.fit(xtrain,ytrain)
  yhat_rf = rf.predict(xtest)
  mse_lwr.append(mse(ytest,yhat_lwr))
  mse_rf.append(mse(ytest,yhat_rf))
print('The MSE for Random Forest is :' + str(np.mean(mse_rf)))
print('The MSE for Locally Weighted Regression is :' + str(np.mean(mse_lwr)))
```

## Dataset 1: 

MSE for each method and which has better results 
final results 

```Python
from sklearn.datasets import load_diabetes
data = load_diabetes()
df = pd.DataFrame(data= np.c_[data['data'], data['target']],
                     columns= data['feature_names'] + ['output'])
x = df['bmi'].values
y = df['output'].values
```

## Dataset 2:

MSE for each method and which has better results 
final results 

```Python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()


df = pd.DataFrame(data= np.c_[data['data'], data['target']],
                     columns= data['feature_names'] + ['output'])

x = df['mean radius'].values
y = df['output'].values
```
