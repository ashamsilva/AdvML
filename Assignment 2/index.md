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
from scipy.interpolate import interp1d #, griddata
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
```


## Dataset 1: 

MSE for each method and which has better results 
final results 

```Python

```

## Dataset 2:

MSE for each method and which has better results 
final results 

```Python

```
