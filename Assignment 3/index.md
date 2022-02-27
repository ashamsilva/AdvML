# Assignment 

Create a new Github page with a presentation on the concepts of Multivariate Regression Analysis and Gradient Boosting. Include a presentation of Extreme Gradient Boosting (xgboost).

Apply the regression methods (including lowess and boosted lowess) to real data sets, such as "Cars" and "Boston Housing Data".  Record the cross-validated mean square errors and the mean absolute errors.
For each method and data set report the crossvalidated mean square error and determine which method is achieveng the better results.
In this paper you should also include theoretical considerations, examples of Python coding and plots. 
The final results should be clearly stated.

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
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
```

Import datasets that will be used in this analysis.
```Python
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer
```

Create the distance function.
```Python
def tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)
```

Create the LOESS Regression Function.
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

Calculate MSE for both Random Forest and Locally Weighted Regression. A for-loop was utilized to determine the optimal tau value. 
```Python
mse_lwr = []
mse_rf = []
rf = RandomForestRegressor(n_estimators=150,max_depth=3)
kf = KFold(n_splits=10,shuffle=True,random_state=1234)
for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
  for idxtrain,idxtest in kf.split(x):
    ytrain = y[idxtrain]
    xtrain = x[idxtrain]
    xtrain = scale.fit_transform(xtrain.reshape(-1,1))
    ytest = y[idxtest]
    xtest = x[idxtest]
    xtest = scale.transform(xtest.reshape(-1,1))
    yhat_lwr = lowess_reg(xtrain.ravel(),ytrain,xtest.ravel(),tricubic,i)
    rf.fit(xtrain,ytrain)
    yhat_rf = rf.predict(xtest)
    mse_lwr.append(mse(ytest,yhat_lwr))
    mse_rf.append(mse(ytest,yhat_rf))
  print('The tau value is :' + str(i))
  print('The MSE for Random Forest is :' + str(np.mean(mse_rf)))
  print('The MSE for Locally Weighted Regression is :' + str(np.mean(mse_lwr)))
```

### Dataset 1: Housing

read in data code 


#### Conclusion:
The Cross-validated Mean Squared Error for LWR is : 54.90110127773951
The Cross-validated Mean Squared Error for BLWR is : 51.35473983508386
The Cross-validated Mean Squared Error for RF is : 51.33316103879288
The Cross-validated Mean Squared Error for XGB is : 41.89036168258404
The Cross-validated Mean Squared Error for NN is : 44.70710846876278
The Cross-validated Mean Squared Error for Nadarya-Watson Regressor is : 45.506043382147325


### Dataset 2: Breast Cancer

Assign the data to a variable and make it into a Pandas dataframe.
```Python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

df = pd.DataFrame(data= np.c_[data['data'], data['target']],
                     columns= list(data.feature_names) + ['output'])
```

Assign the input and output variables to x and y.
```Python
x = df['mean radius'].values
y = df['output'].values
```

#### Conclusion:
The minimum mean squared error for both Locally Weighted Regression and Random Forest were found with a tau value of 0.1. 
The MSE for Random Forest is :0.09312817955310984
The MSE for Locally Weighted Regression is :0.09020271929160795
