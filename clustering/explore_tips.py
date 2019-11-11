"""
Create either a python script or a jupyter notebook named `explore_tips` that explored the tips data set that is built in to seaborn.
Perform at least 1 t-test, and 1 chi square test.
"""


import seaborn as sns
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings("ignore")

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from math import sqrt
import scipy.stats as stats

import matplotlib
plt.rc('font', size=14)
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import learning_curve
#%matplotlib inline

#pd.options.display.float_format = '{:20, .2f}'.format

# Load the dataset:
df = sns.load_dataset('tips')

# Explore data types, and statistics of tips
df.info()

df.describe()

# Create histogram plot
df.hist(color='black')

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

labels = ['Dinner', 'Lunch']
no_smoker_counts = list(observed.no_smoker)
yes_smoker_counts = list(observed.yes_smoker)

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rect1 = ax.bar(x-width/2, no_smoker_counts, width, label = 'no_smoker')
rect2 =ax.bar(x+width/2, yes_smoker_counts, width, label = 'yes_smoker')

ax.set_ylabel('count')
ax.set_title('Count of Smokers by Time')
ax.set_xticks(x)
ax.set_xticklabels(labels)
# With a higher p value of .47 we can say that smoking is independent of time of day 
def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')

autolabel(rect1, "center")
autolabel(rect2, "center")


