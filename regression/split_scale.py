""" Our scenario continues:
    As a customer analyst, I want to know who has spent the most money with us over their lifetime.
    I have monthly charges and tenure, so I think I will be able to us those attributes as features to estimate total_charges.
    I need to do this within an average of $5.00 per customer.
    
    Create a split_scale.py that will contain the functions that follow.
    Each scaler function should create an object, fit and transform both train and test.
    They should return a scaler, train dataframe scaled, test dataframe scaled.
    Be sure your indices represent the original indicies from the original dataframe.
    Be sure to set a random state where applicable for reproducibility! """

# 1. split_my_data(X,y, train_pct)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler


import wrangle
import env

# acquire data and remove null values.
# df = wrangle.wrangle_telco()

# verify acquisition
# df.info()

# df.describe()

# Create dataframes for independent variables (X) and target variables (y)

#X = df.drop(columns = 'total_charges').set_index('customer_id')
#y = pd.DataFrame({'total_charges': df.total_charges, 'customer_id': df.customer_id}).set_index('customer_id')

# 1. split_my_data(X)
def split_my_data(X):
    """ Input dataframes for X and y  \nParameters:  \nX - independent variables  \ny - target variables  \ntrain_pct - percent for test/train split"""
    train, test = train_test_split(X, train_size =.80, random_state = 123)
    return train, test

#X_train, X_test = split_my_data(X)
#y_train, y_test = split_my_data(y)


# X_train set_index to "customer_id"

# 2. standard_scalar()

def standard_scaler(X):
    train, test =  split_my_data(X)
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)
    train_scaled_data = pd.DataFrame(scaler.transform(train),columns=train.columns.values).set_index([train.index.values])
    test_scaled_data = pd.DataFrame(scaler.transform(test),columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled_data, test_scaled_data

# X_scaler, X_train_scaled_data, X_test_scaled_data = standard_scaler(X)

# y_scaler, y_train_scaled_data, y_test_scaled_data = standard_scaler(y)


# Look at the mean and standard deviation that will be used to transform the data.
#import math

# X_scalar.mean_
# [math.sqrt(i) for i in X_scalar.var_]

#sns.pairplot(X_train_scaled_data, kind = "reg")

# y_scalar.mean_
# y_train.mean()
# [math.sqrt(i) for i in y_scalar.var_] (standard deviation)
# y_train.std()


# 3. scale_inverse()

def scaler_inverse(X):
    scaler, train_scaled_data, test_scaled_data = standard_scaler(X)
    train_unscaled = pd.DataFrame(scaler.inverse_transform(train_scaled_data), columns = train_scaled_data.columns.values).set_index([train_scaled_data.index.values])
    test_unscaled = pd.DataFrame(scaler.inverse_transform(test_scaled_data), columns = test_scaled_data.columns.values).set_index([test_scaled_data.index.values])
    return scaler, train_unscaled, test_unscaled

# X_scaler, X_train_unscaled, X_test_unscaled = scaler_inverse(X)
# y_scaler, y_train_unscaled, y_test_unscaled = scaler_inverse(y)

# 4. uniform_scaler()

def uniform_scaler(X):
    train, test = split_my_data(X)
    scaler = QuantileTransformer(n_quantiles=100, output_distribution = 'uniform', random_state = 123, copy = True).fit(train)
    train_scaled_data = pd.DataFrame(scaler.transform(train), columns = train.columns.values).set_index([train.index.values])
    test_scaled_data = pd.DataFrame(scaler.transform(test),columns = test.columns.values).set_index([test.index.values])
    return scaler, train_scaled_data, test_scaled_data

# X_scaler, X_train_scaled_data, X_test_scaled_data = uniform_scaler(X)
# y_scaler, y_train_scaled_data, y_test_scaled_data = uniform_scaler(y)

def gaussian_scaler(X):
    train, test = split_my_data(X)
    scaler = PowerTransformer(method = 'box-cox', standardize = False, copy = True).fit(train)
    train_scaled_data = pd.DataFrame(scaler.transform(train), columns = train.columns.values).set_index([train.index.values])
    test_scaled_data = pd.DataFrame(scaler.transform(test), columns = test.columns.values).set_index([test.index.values])
    return scaler, train_scaled_data, test_scaled_data

# X_scaler, X_train_scaled_data, X_test_scaled_data = gaussian_scaler(X)
# y_scaler, y_train_scaled_data, y_test_scaled_data = gaussian_scaler(y)

# min_max_scaler()
def min_max_scaler(X):
    test, train = split_my_data(X)
    scaler = MinMaxScaler(copy = True, feature_range = (0,1)).fit(train)
    train_scaled_data = pd.DataFrame(scaler.transform(train), columns = train.columns.values).set_index([train.index.values])
    test_scaled_data = pd.DataFrame(scaler.transform(test), columns = test.columns.values).set_index([test.index.values])
    return scaler, train_scaled_data, test_scaled_data

# X_scaler, X_train_scaled_data, X_test_scaled_data = min_max_scaler(X)
# y_scaler, y_train_scaled_data, y_test_scaled_data = min_max_scaler(y)

def iqr_robust_scaler(X):
    train, test = split_my_data(X)
    scaler = RobustScaler(quantile_range = (25.0,75.0),copy = True, with_centering = True, with_scaling = True).fit(train)
    train_scaled_data = pd.DataFrame(scaler.transform(train), columns = train.columns.values).set_index([train.index.values])
    test_scaled_data = pd.DataFrame(scaler.transform(test), columns = test.columns.values).set_index([test.index.values])
    return scaler, train_scaled_data, test_scaled_data

# X_scaler, X_train_scaled_data, X_test_scaled_data = iqr_robust_scaler(X)
# y_scaler, y_train_scaled_data, y_test_scaled_data = iqr_robust_scaler(y)

