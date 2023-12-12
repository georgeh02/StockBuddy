"""
Author: George Harrison
Date: 12-05-23
Description: script for training models on dataset with stock symbol feature
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, RidgeCV, ElasticNetCV, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

#df = pd.read_csv('panel_step13_500stocks.csv')
df = pd.read_csv('panel_step14_1000stocks.csv')

## PREPROCESSING--------------------------------------------------------
df = df.drop(columns = 'URL')
df = df.drop(columns = 'end_prices') # this column will be blank in my real tests

# dropping due to multicollinearity and redundancy
df = df.drop(columns = 'sharesBeforeTransaction')

# dates
df = df.drop(columns = 'periodOfReport')
df = df.drop(columns = 'end_dates')


X = df.drop('percentPriceChange', axis=1)
y = df['percentPriceChange']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## TRAINING-----------------------------------------------------------------
def train_models(model, X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Trains and evaluates a model

    Parameters
    --------------------
        model             -- model with tuned parameters
        X_train_scaled    -- scaled training datasplit
        y_train           -- y training datasplit
        X_test_scaled     -- scaled testing datasplit
        y_test            -- y testing datasplit
    """

    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print(f"{model.__class__.__name__} MSE (train): {round(mse_train, 3)}")
    print(f"{model.__class__.__name__} MSE (test): {round(mse_test, 3)}")
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    print(f"{model.__class__.__name__} MAE (train): {round(mae_train, 3)}")
    print(f"{model.__class__.__name__} MAE (test): {round(mae_test, 3)}")
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    print(f"{model.__class__.__name__} R^2 (train): {round(r2_train, 3)}")
    print(f"{model.__class__.__name__} R^2 (test): {round(r2_test, 3)}")
    directional_accuracy_train = (y_train * y_train_pred > 0).mean()
    directional_accuracy_test = (y_test * y_test_pred > 0).mean()
    print(f"{model.__class__.__name__} Directional Accuracy (train): {round(directional_accuracy_train, 3)}")
    print(f"{model.__class__.__name__} Directional Accuracy (test): {round(directional_accuracy_test, 3)}")

## OPTIMIZING RIDGE
ridge_cv_model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
ridge_cv_model.fit(X_train_scaled, y_train)
best_alpha = ridge_cv_model.alpha_
print("Ridge Best Alpha:", best_alpha)

## OPTIMIZING ELASTICNET
elasticnet_cv_model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], cv=5)
elasticnet_cv_model.fit(X_train_scaled, y_train)
best_alpha = elasticnet_cv_model.alpha_
best_l1_ratio = elasticnet_cv_model.l1_ratio_
print("ElasticNet Best Alpha:", best_alpha)
print("ElasticNet Best L1 Ratio:", best_l1_ratio)

## OPTIMIZING SGD
param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
grid_search = GridSearchCV(estimator=SGDRegressor(), param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
best_alpha = grid_search.best_params_['alpha']
best_l1_ratio = grid_search.best_params_['l1_ratio']
print("SGD Best Alpha:", best_alpha)
print("SGD Best L1 Ratio:", best_l1_ratio)
sgd_best = grid_search.best_estimator_


ridge = train_models(Ridge(alpha=best_alpha), X_train_scaled, y_train, X_test_scaled, y_test)
elasticnet = train_models(ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio), X_train_scaled, y_train, X_test_scaled, y_test)
sgd = train_models(sgd_best, X_train_scaled, y_train, X_test_scaled, y_test)

# Save the trained model to a file
for model in [ridge, elasticnet, sgd]:
    joblib.dump(model, f'{model.__class__.__name__}.joblib')


# Get feature coefficients
feature_coefficients = list(zip(X.columns, sgd.coef_))
for feature, coefficient in feature_coefficients:
    print(f"{feature}: {coefficient}")


# ## OPTIMIZING SVR
# param_grid = {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1, 1]}
# grid_search = GridSearchCV(SVR(kernel='linear'), param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X_train_scaled, y_train)
# best_c = grid_search.best_params_['C']
# best_epsilon = grid_search.best_params_['epsilon']
# print("SVR Best C:", best_c)
# print("SVR Best Epsilon:", best_epsilon)
# svr_best = grid_search.best_estimator_

# svr = train_models(svr_best, X_train_scaled, y_train, X_test_scaled, y_test)