"""
Create either a python script or a jupyter notebook named `explore_tips` that explored the tips data set that is built in to seaborn.
Perform at least 1 t-test, and 1 chi square test.
"""


import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from math import sqrt
import scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import learning_curve
#%matplotlib inline

#pd.options.display.float_format = '{:20, .2f}'.format

# You can 
df = sns.load_dataset('tips')


df.info()

df.describe()

df.hist()
sns.boxplot(data=df)

sns.jointplot(x='tip',y='total_bill', data=df)
sns.jointplot(x='tip', y= 'size', data=df)
sns.jointplot(x='total_bill', y = 'size', data=df)

sns.pairplot(df, hue='day', palette = 'husl')

sns.heatmap(df.corr(), annot=True)

df.groupby('smoker').mean().plot.bar()

# T-test:
# H[0]: There is no difference in the tip amount for customers who are male and customers who are female.
# H[1]: There is a difference in the tip amount for customers who are male and customers who are female.

x1 = df[df.sex == 'Male'].tip
x2 = df[df.sex == 'Female'].tip

p_value = stats.ttest_ind(x1,x2)[1]

# Because the p_value is > alpha=.05 we can accept the null hypothesis that there is not a statistically significant difference in the average tip amount of males and females between

df[df.sex == 'Male'].tip.mean()
df[df.sex == 'Female'].tip.mean()

# Chi^2 test
# H[0]: Smoking is independent of time of day a customer come to the restaurant

no_smoker = df[['smoker','time']].time[df.smoker == 'No'].value_counts()
yes_smoker = df[['smoker','time']].time[df.smoker == 'Yes'].value_counts()

observed = pd.DataFrame({'no_smoker': no_smoker, 'yes_smoker': yes_smoker})

stats.chi2_contingency(observed)

# With a higher p value of .47 we can say that smoking is independent of time of day 
observed.boxplot()