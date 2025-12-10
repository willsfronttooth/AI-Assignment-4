# src/regression_models.py
"""
Regression model helpers: linear and polynomial.
"""
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def train_linear(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_polynomial(X_train, y_train, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    return model, poly

def evaluate(model, X_test, y_test, poly=None):
    if poly is not None:
        X_test_trans = poly.transform(X_test)
    else:
        X_test_trans = X_test
    preds = model.predict(X_test_trans)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {'mse': mse, 'mae': mae, 'r2': r2, 'y_pred': preds}