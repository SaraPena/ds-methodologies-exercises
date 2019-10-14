""" Our scenario continues:
    As a customer analyst , I want to know who has spent the most money wit hus over their lifetime.
    I have monthly charges, and tenure so I think I will be able to use those two attributes as features to estimate total_charges.
    I need to do this within an average of $5.00 per customer."""

# Create a file, explore.py, that contains the following functions for exploring your variables. (features & target).

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import pandas as pd 
import numpy as numpy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

import env
import wrangle
import split_scale

df = wrangle.wrangle_telco()

X = df[['tenure','monthly_charges','total_charges']]

X_train, x_test = split_scale.split_my_data(X)

# 1. Write a function, plot_variable_pairs(dataframe) that plots all of the pairwise relationships along with the regression line for each pair.

def plot_variable_pairs(df):
    train, test = train_test_split(df)
    return sns.pairplot(data = train, kind = 'reg'), sns.pairplot(data = test, kind = 'reg')

plot_variable_pairs(X)

# 2. Write a function, months_to_years(tenure_months,df) that returns your dataframe with a new feature tenure_years, in complete years as a customer.
def months_to_years(tenure_month,df):
    df['tenure_years'] = round(tenure_month / 12)
    return df

months_to_years(df.tenure,df)

# 3. Write a function, plot_categorical_and_continuous_vars(categorical_var,continuous_var,df), that outputs 3 different plots for plotting a categorical variable with a continuous variable, e.g. tenure_years with total_charges.
#    For ideas on effective ways to visualize categorical with continuous: <https://datavizcatalogue.com/>.
#    You can then look into seaborn and matplotlib documentation for ways to create plots.

def plot_categorical_and_continuous_vars(categorical_var,continuous_var1,continous_var2,continuous_var3, df):
    f, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
    sns.boxplot(x=categorical_var, y=continuous_var1, data=df, ax=axes[0])
    sns.scatterplot(x=continous_var2, y=continuous_var1, hue=categorical_var, data=df, ax=axes[1])
    sns.stripplot(x=continuous_var3, y=continuous_var1, data=df, ax=axes[2])
    plt.show

plot_categorical_and_continuous_vars('internet_service_type', 'total_charges','tenure','tenure_years',df)
