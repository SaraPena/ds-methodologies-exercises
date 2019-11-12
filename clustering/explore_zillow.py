"""
Zillow

Create a python script or jupyter notebook named `explore_zillow` that does the following:
"""

# ignore warnings
import warnings

# Wrangling
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Exploring
import scipy.stats as stats

# Visualizing
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
sns.set_style('whitegrid') 

import acquire
import summarize
import prepare
import wrangle_zillow

# Use wrangle_zollow to bring in zillow dataset
df = wrangle_zillow.wrangle_zillow_data()

# Look at datatypes of zillow
df.info()

# Look at fips variable
df.fips.value_counts()

# drop the id column and set the parcelid to be the index of zillow
df.drop(columns = ['id'], inplace = True)
df.set_index('parcelid', inplace = True)
df. head()

# Split data:
train, test = train_test_split(df, test_size = .3, random_state = 42)

# Use MinMaxScaler to scale variables that are number types.
scaler = MinMaxScaler()
num_vars = list(train.select_dtypes('number').columns)
train[num_vars] = scaler.fit_transform(train[num_vars])

train.select_dtypes('number').hist(figsize = (16,12), bins = 5, color = 'blue')

sns.pairplot(train)

plt.figure(figsize = (24,8))
sns.boxplot(data=train)
sns.boxplot(train.logerror)
sns.distplot(train.logerror)
plt.xlim(0.3, 0.6)

plt.figure(figsize = (24,8))
sns.heatmap(train.select_dtypes('number').corr(), annot=True)


train.corr()
train.fips.value_counts()
df.fips.value_counts()

sns.scatterplot(x='longitude', y = 'latitude', data = train, hue = 'fips')

train.logerror.hist()

sns.relplot(x='calculatedfinishedsquarefeet', y='logerror', data=train, hue='fips')

train['fips'] = train.fips.astype('str')
train.info()

df.logerror.hist()
df.calculatedfinishedsquarefeet.hist()
sns.relplot(x='longitude', y = 'logerror', data=df, hue='fips')
sns.scatterplot(x='latitude', y = 'logerror', data = df, hue ='fips')
sns.scatterplot(x='lotsizesquarefeet', y='logerror', data=df,hue='fips')
df['fips'] = df.fips.astype('float')

df.fips.dtype
df.dtypes


sns.scatterplot(x='taxvaluedollarcnt', y = 'logerror', data=df, hue='fips')
g = sns.pairplot(df[['logerror', 'calculatedfinishedsquarefeet']])