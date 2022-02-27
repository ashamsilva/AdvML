# Assignment 

Create a new Github page with a presentation on the concepts of Multivariate Regression Analysis and Gradient Boosting. Include a presentation of Extreme Gradient Boosting (xgboost).

Apply the regression methods (including lowess and boosted lowess) to real data sets, such as "Cars" and "Boston Housing Data".  Record the cross-validated mean square errors and the mean absolute errors.
For each method and data set report the crossvalidated mean square error and determine which method is achieveng the better results.
In this paper you should also include theoretical considerations, examples of Python coding and plots. 
The final results should be clearly stated.


# what I did for the last one

Locally Weighted Regression is a specialized type of regression which utilizes weighted linear regression to make more accurate predictions. LO(W)ESS or LOESS is non-parametric. LOESS calculates a predicted regression line by isolating neighboring points within a specified range and calculate an estimate.

In order to find the k nearest neighbors from x, Euclidean distance is used. The distance calculated is then utilized to find the weights for the regression. In the code section below, the execution of this can be found within the tricubic() function. In this process, weighting works by giving more meaning and greater weight the closer a point is to the x value. For example - a point with zero distance will be given a weight of one.

put equation here

In conclusion, the predictions we make are a linear combination of the actual observed values of the dependent variable and by using locally weighted regression we obtained the predicted y as a different linear combination of the values of y.
#end what i did for the last one

# Multivariate Regression Analysis and Gradient Boosting

### Multivariate Regression Analysis

Multivariate regression analysis is a more robust continuation of a linear regression analysis and has more than one independent variable. Unlike linear regressions, we do not have to make assumptions regarding predictors. Linear regressions which define y as a function of x. Multivariate regression analysis allows us to define y as a function of multiple variables (ie. x and z) such that y = f(x,z). In adding this additional variable, an additional dimension is added as well. In the example that y is a function of x and z a plane would be predicted. 

The equation for the predictions we make is:
y = B0 + B1*X1 + B2*X2 + ... + Bn+Xn


### Gradient Boosting

Gradient boosting, or gbm, is a method used in machine learning to increase the stregth of learners. When gradient boosting is used, each additional tree is fit on a version of the orginal dataset that has been modified. Using these decision trees, the model is trained and produces a final prediction value of the dependent variable. This process takes the mean from the original dataset and adds the residuals predicted by the decision trees in the forest.








#### Extreme Gradient Boosting (xgboost)

