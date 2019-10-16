""" 
Our scenario continues:
As a customer analyst, I want to know who has spent the most money with us over their lifetime.
I have monthly charges and tenure, so I think I will be able to use those attributes as features to estimate total_charges.
I need to do this within an average of $5.00 per customer.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

import env
import wrangle
import split_scale

# acquire data and remove null values

df = wrangle.wrangle_telco()
df.dtypes

df = df[['monthly_charges', 'tenure', 'total_charges']]

train, test = split_scale.split_my_data(df)

# split into train and test
# For feature engineering methods, we want to use the scaled data:
# scale the data using standard scaler

scaler, train_scaled_data, test_scaled_data = split_scale.standard_scaler(df)

# to return to orignal values
# scaler, train, test = split_scale.scaler_inverse(df)

X_train = train.drop(columns = 'total_charges')
y_train = train[['total_charges']]

X_test = test.drop(columns = 'total_charges')
y_test = test[['total_charges']]

# Using Pearson Correlation

sns.set_style('whitegrid')
plt.figure(figsize=(6,5))
cor = train.corr()
sns.heatmap(cor, annot = True, cmap=plt.cm.Reds)

# To view just the correlations of each attribute with the target variable, and filter down to only those above a certain value:
# Correlation with output variable
cor_target = abs(cor["total_charges"])

#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]

# Looking at the correlation of each of these variables, monthly_charges is most correlated with total_charges. Tenure and monthly_charges are correlated with each other.
# So using this method, we may decide to only use monthly_charges or we may want to create a new variable that uses monthly_charges and tenure.
# Would need to scale monthly_charges, and tenure since they are measured differently (money vs. time). calculate a proportion of monthly_charges to tenure? monthly_to_tenure = monthly_charges/tenure

# 1. Write a function, select_kbest_freg_unscaled(X_train,y_train,k) that takes X_train, y_train and k as input (X_train and y_train should not be scaled!) and returns a list of the top k features

from sklearn.feature_selection import SelectKBest, f_regression

def select_k_best_freg_unscaled(X_train,y_train,k):
    f_selector = SelectKBest(f_regression,k=k)
    
    f_selector.fit(X_train,y_train)

    f_support = f_selector.get_support()
    f_feature = X_train.loc[:,f_support].columns.tolist()
    return f_feature

# select_k_best_freg_unscaled(X_train,y_train,2)

# 2. Write a function, select_kbest_freg() that takes X_train, y_train(scaled) and k as in put and returns a list of the top k features.

def select_kbest_freg_scaled(X_train, y_train, k):
    X_scaler, X_train_scaled_data, X_test_scaled_data = split_scale.standard_scaler(X_train)
    y_scaler, y_train_scaled_data, y_test_scaled_data = split_scale.standard_scaler(y_train)

    f_selector = SelectKBest(f_regression,k=k)
    f_selector.fit(X_train_scaled_data, y_train_scaled_data)

    f_support = f_selector.get_support()
    f_feature = X_train_scaled_data.loc[:,f_support].columns.tolist()
    return f_feature

# select_kbest_freg_scaled(X_train, y_train, 2)

# 3. Write a function, ols_backward_elimination() that takes X_train and y_train(scaled) as input and returns selected features based on the ols backwards elimation method.

import statsmodels.api as sm

def ols_backward_elimination(X_train, y_train):
    cols = list(X_train.columns)
    pmax = 1
    while (len(cols)>0):
        model = sm.OLS(y_train, X_train).fit()
        p = pd.Series(model.pvalues.values[0:], index = cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax>0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols
    return selected_features_BE

#ols_backward_elimination(X_train,y_train)

#ols_backward_elimination(X_train_scaled_data,y_train_scaled_data)

# 4. Write a function, lasso_cv_coef() that takes X_train and y_train as input and returns coefficients for each feature, along with a plot of the the features and their weights.

from sklearn.linear_model import LassoCV
import matplotlib

def lasso_cv_coef(X_train, y_train):
    reg = LassoCV()
    reg.fit(X_train,y_train)
    coef = pd.Series(reg.coef_, index = X_train.columns)
    imp_coef = coef.sort_values()
    matplotlib.rcParams['figure.figsize'] = (4.0, 5.0)
    imp_coef.plot(kind ='barh')
    plt.title("Feature importance using Lasso Model")
    return imp_coef

lasso_cv_coef(X_train_scaled_data, y_train_scaled_data)

# 5. Write 3 functions, the first computes the number of optimum features (n) using rfe, the second takes n as input and returns the top n features, and the third takes the list of the top n features as input and returns a new X_train, and X_test dataframe with those top features, recursive_feature_elimination() that computes the optimum number of features (n) and returns the top n features.
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

def recursive_feature_elimination(n, X_train,y_train,y_test):
    # Initializing RFE model, with parameter to select top 2 features.
    number_of_features_list = np.arange(1,n+1)
    high_score = 0

    #Vairable to store the optimum features
    number_of_features = 0
    score_list  = []
    for n in range(len(number_of_features_list)):
        model = LinearRegression()
        rfe = RFE(model,number_of_features_list[n])
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)
        if score > high_score:
            high_score = score
            number_of_features = number_of_features_list[n]
        return number_of_features, high_score

recursive_feature_elimination(2, X_train, y_train, y_test)

def recursive_feature_selection(number_of_features)
    cols = list(X_train.columns)
    model = LinearRegression()
    rfe = RFE(model,1)
    X_rfe = rfe.fit_transform(X_train,y_train)
    model.fit(X_rfe,y_train)
    temp = pd.Series(rfe.support_)
