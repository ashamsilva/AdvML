# Assignment 5

Import the needed libraries
```Python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error as MSE
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator 
import warnings
from numba import jit, prange
from scipy.optimize import minimize
```

## Question 1: 

SCAD
```Python
def scad_penalty(beta_hat, lambda_val, a_val):
    is_linear = (np.abs(beta_hat) <= lambda_val)
    is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
    is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
    linear_part = lambda_val * np.abs(beta_hat) * is_linear
    quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
    constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant
    return linear_part + quadratic_part + constant_part
    
def scad_derivative(beta_hat, lambda_val, a_val):
    return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))

class SCAD(BaseEstimator):
    def __init__(self, maxiter=12000, lam = 0.001, alpha=2):
      self.maxiter = maxiter
      self.alpha = alpha
      self.lam = lam

    def fit(self, X, y): 
      # we add aan extra columns of 1 for the intercept
      #X = np.c_[np.ones((n,1)),X]
      n = X.shape[0]
      p = X.shape[1]
      def scad(beta):
        beta = beta.flatten()
        beta = beta.reshape(-1,1)
        n = len(y)
        return 1/n*np.sum((y-X.dot(beta))**2) + np.sum(scad_penalty(beta,self.lam,self.alpha))
      
      def dscad(beta):
        beta = beta.flatten()
        beta = beta.reshape(-1,1)
        n = len(y)
        return np.array(-2/n*np.transpose(X).dot(y-X.dot(beta))+scad_derivative(beta,self.lam,self.alpha)).flatten()
      b0 = np.ones((p,1))
      output = minimize(scad, b0, method='L-BFGS-B', jac=dscad,options={'gtol': 1e-8, 'maxiter': 1e7,'maxls': 25,'disp': True})
      self.coef = output.x
      return output.x
    
    def predict(self, x): 
      return x.dot(self.coef)

    def get_params(self, deep=True):
    # suppose this estimator has parameters "alpha" and "recursive"
      return {"lam": self.lam, 'a': self.alpha}

    def set_params(self, **parameters):
      for parameter, value in parameters.items():
          setattr(self, parameter, value)
      return self
```

SQRT Lasso
```Python
class SQRTLasso(BaseEstimator): #, RegressorMixin):
    def __init__(self, maxiter = 12000, alpha=0.01):
        self.maxiter = maxiter
        self.alpha = alpha

    def fit(self, x, y):
        alpha=self.alpha
        def f_obj(x,y,beta,alpha):
          n =len(x)
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          output = np.sqrt(1/n*np.sum((y-x.dot(beta))**2)) + alpha*np.sum(np.abs(beta))
          return output
        def f_grad(x,y,beta,alpha):
          n=x.shape[0]
          p=x.shape[1]
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          output = np.array((-1/np.sqrt(n))*np.transpose(x).dot(y-x.dot(beta))/np.sqrt(np.sum((y-x.dot(beta))**2))+alpha*np.sign(beta)).flatten()
          return output
        def objective(beta):
          return(f_obj(x,y,beta,alpha))
        def gradient(beta):
          return(f_grad(x,y,beta,alpha))
        
        beta0 = np.ones((x.shape[1],1))
        output = minimize(objective, beta0, method='L-BFGS-B', jac=gradient,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 25,'disp': True})
        beta = output.x
        self.coef_ = beta
        
    def predict(self, x):
        return x.dot(self.coef_)
```

## Question 2:

```Python
n = 200
p = 1200
rho = 0.8
beta_star = np.concatenate(([1]*7,[0]*25,[0.25]*5,[0]*50,[0.7]*15,[0]*1098))

v = []
for i in range(p):
  v.append(rho**i)

r = toeplitz(v)
mu = [0]*p
sigma = 3.5

# Generate the random samples.
np.random.seed(123)
x = np.random.multivariate_normal(mu, toeplitz(v), size=n) # this where we generate some fake data
y = np.matmul(x,beta_star).reshape(-1,1) + sigma*np.random.normal(0,1,size=(n,1))

# what we want to detect is the position of the actual information or "signal"
pos = np.where(beta_star != 0)
```
## Question 3:

Ridge 
```Python
ridge_reg = Ridge()
params = [{'alpha':np.linspace(0.001,10,num=50)}]
gs = GridSearchCV(estimator=ridge_reg,cv=10,scoring='neg_mean_squared_error',param_grid=params)
gs_results = gs.fit(x,y)
print(gs_results.best_params_)

kf = KFold(n_splits=10, shuffle=True, random_state=123)
coeffs = []
rmse = []
l2_dist = []
alpha = 0.001

for train_index , test_index in kf.split(x):
    x = pd.DataFrame(x)
    x_train , x_test = x.iloc[train_index,:].values,x.iloc[test_index,:].values
    y_train, y_test = y[train_index], y[test_index]
    model = Ridge(alpha = alpha)
    model.fit(x_train, y_train)
    beta_hat = model.coef_
    pos_lasso = np.where(beta_hat != 0)
    coeffs.append(len(np.intersect1d(pos, pos_lasso)))
    yhat = model.predict(x_test)
    rmse.append(MSE(yhat, y_test)**.5)
    l2_dist.append(np.linalg.norm(model.coef_ - beta_star,ord = 2))
print("Average true non-zero coefficients:", np.mean(coeffs))
print("Average RMSE:", np.mean(rmse))
print("Average L2 Distance to Ideal:", np.mean(l2_dist))
```
Average true non-zero coefficients: 27.0
Average RMSE: 5.993548577298991
Average L2 Distance to Ideal: 3.0194654516702113

Lasso
```Python
model = Lasso()
with warnings.catch_warnings():
  warnings.simplefilter("ignore")

  lasreg = Lasso()
  params = [{'alpha':np.linspace(0.001,2,num=50)}]
  gs = GridSearchCV(estimator=lasreg,cv=10,scoring='neg_mean_squared_error',param_grid=params)
  gs_results = gs.fit(x,y)
  print(gs_results.best_params_)
  print('The mean square error is: ', np.abs(gs_results.best_score_))
  
kf = KFold(n_splits=10, shuffle=True, random_state=2021)
coeffs = []
rmse = []
l2_dist = []
alpha = 0.001
for train_index , test_index in kf.split(x):
    x = pd.DataFrame(x)
    x_train , x_test = x.iloc[train_index,:].values,x.iloc[test_index,:].values
    y_train, y_test = y[train_index], y[test_index]
    model = Lasso(alpha = alpha)
    model.fit(x_train, y_train)
    beta_hat = model.coef_
    pos_lasso = np.where(beta_hat != 0)
    coeffs.append(len(np.intersect1d(pos, pos_lasso)))
    yhat = model.predict(x_test)
    rmse.append(MSE(yhat, y_test)**.5)
    l2_dist.append(np.linalg.norm(model.coef_ - beta_star,ord = 2))
print("Average true non-zero coefficients:", np.mean(coeffs))
print("Average RMSE:", np.mean(rmse))
print("Average L2 Distance to Ideal:", np.mean(l2_dist))
```
Average true non-zero coefficients: 23.8
Average RMSE: 4.010829295726692
Average L2 Distance to Ideal: 3.6606059802920043

Elastic Net
```Python
ENreg = ElasticNet(max_iter=1200)
params = [{'alpha':np.linspace(0.001,1,num=50),'l1_ratio':np.linspace(0,1,num=50)}]
gs = GridSearchCV(estimator=ENreg,cv=10,scoring='neg_mean_squared_error',param_grid=params)
gs_results = gs.fit(x,y)
print(gs_results.best_params_)

kf = KFold(n_splits=10, shuffle=True, random_state=2021)
coeffs = []
rmse = []
l2_dist = []
alpha = 0.001
l1_ratio = 0.4897959183673469
for train_index , test_index in kf.split(x):
    x = pd.DataFrame(x)
    x_train , x_test = x.iloc[train_index,:].values,x.iloc[test_index,:].values
    y_train, y_test = y[train_index], y[test_index]
    model = ElasticNet(alpha = alpha, l1_ratio = l1_ratio)
    model.fit(x_train, y_train)
    beta_hat = model.coef_
    pos_lasso = np.where(beta_hat != 0)
    coeffs.append(len(np.intersect1d(pos, pos_lasso)))
    yhat = model.predict(x_test)
    rmse.append(MSE(yhat, y_test)**.5)
    l2_dist.append(np.linalg.norm(model.coef_ - beta_star,ord = 2))
print("Average true non-zero coefficients:", np.mean(coeffs))
print("Average RMSE:", np.mean(rmse))
print("Average L2 Distance to Ideal:", np.mean(l2_dist))
```
Average true non-zero coefficients: 25.6
Average RMSE: 3.8884725670875446
Average L2 Distance to Ideal: 3.0824983981417753

SCAD
```Python
model = SCAD()
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  scad_params = [{'lam': np.linspace(0.001, 1, 25), 'a': np.linspace(.1, 3, 25)}]

  grid = GridSearchCV(model, scad_params, cv = 5, scoring='neg_mean_absolute_error')

  grid.fit(X, y)
  
  
kf = KFold(n_splits=10, shuffle=True, random_state=2021)
coeffs = []
rmse = []
l2_dist = []
for train_index , test_index in kf.split(X):
    X = pd.DataFrame(X)
    X_train , X_test = X.iloc[train_index,:].values,X.iloc[test_index,:].values
    y_train, y_test = y[train_index], y[test_index]
    model = SCAD(lam = grid.best_params_.get('lam'), a = grid.best_params_.get('a'))
    model.fit(X_train, y_train)
    beta_hat = model.coef
    pos_lasso = np.where(beta_hat != 0)
    coeffs.append(len(np.intersect1d(pos, pos_lasso)))
    yhat = model.predict(X_test)
    rmse.append(MSE(yhat, y_test)**.5)
    l2_dist.append(np.linalg.norm(model.coef - beta_star,ord = 2))
print("Average true non-zero coefficients:", np.mean(coeffs))
print("Average RMSE:", np.mean(rmse))
print("Average L2 Distance to Ideal:", np.mean(l2_dist))
```
Average true non-zero coefficients: 27.0
Average RMSE: 9.069864242200328
Average L2 Distance to Ideal: 6.343565875213969

SQRT Lasso
```Python
def validate(model, x, y, nfolds = 10, rs = 123):
  kf = KFold(n_splits = nfolds, shuffle = True, random_state = rs)
  # prediction error
  PE = []
  for idxtrain, idxtest in kf.split(x):
    xtrain = x[idxtrain]
    ytrain = y[idxtrain]
    xtest = x[idxtest]
    ytest = y[idxtest]
    model.fit(xtrain, ytrain)
    PE.append(mean_absolute_error(ytest, model.predict(xtest)))
  return np.mean(PE)
  
a_range = np.linspace(0,0.5,50)
test_mae = []
for a in a_range:
  model = SQRTLasso(alpha = a)
  pe = validate(model, x, y)
  print(pe)
  test_mae.append(pe)
  
d = {"alpha": a_range, "mse": test_mae}
df = pd.DataFrame(d)

test_mae[np.argmin(test_mae)]
df.loc[df['mse'] == 3.039775937024018, 'alpha']

kf = KFold(n_splits=10, shuffle=True, random_state=123)
coeffs = []
rmse = []
l2_dist = []
alpha = 0.153061
for train_index , test_index in kf.split(x):
    x = pd.DataFrame(x)
    x_train , x_test = x.iloc[train_index,:].values,x.iloc[test_index,:].values
    y_train, y_test = y[train_index], y[test_index]
    model = SQRTLasso(alpha = alpha)
    model.fit(x_train, y_train)
    beta_hat = model.coef_
    pos_lasso = np.where(beta_hat != 0)
    coeffs.append(len(np.intersect1d(pos, pos_lasso)))
    yhat = model.predict(x_test)
    rmse.append(MSE(yhat, y_test)**.5)
    l2_dist.append(np.linalg.norm(model.coef_ - beta_star,ord = 2))
print("Average true non-zero coefficients:", np.mean(coeffs))
print("Average RMSE:", np.mean(rmse))
print("Average L2 Distance to Ideal:", np.mean(l2_dist))
```
Average true non-zero coefficients: 27.0
Average RMSE: 3.9151925764275965
Average L2 Distance to Ideal: 1.3281744764820536
