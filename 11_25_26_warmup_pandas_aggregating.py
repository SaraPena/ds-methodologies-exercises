# Pandas Aggregating Warmup

# 0. Do you work in whatever format you feel comforatble with (notebook, repl, etc.).
#    Get the data from the_data.csv into a data frame(you can click the "Raw button in the upper left corner to download the data").
# 1. Make sure timestamp is a datatime type and set it as the index.

import pandas as pd
import numpy as np

df = pd.read_csv('the_data.csv')
df.head()
df.info()

df['timestamp'] = pd.to_datetime(df.timestamp)
df.info()

df.sort_values(by='timestamp', inplace=True)

df.set_index('timestamp', inplace = True)


# 2. Answer the questions below:

# 1. By Group:
# - What is the maximum x value for group B

df[df.y == 'B'].x.max()

df.groupby(df.y)['x'].agg(['mean','min','max']).loc['B']['max']


# - What is the average x value for group A
df[df.y == 'A'].x.mean()

df.groupby(df.y)['x'].agg(['mean','min','max']).loc['A']['mean']


# - What is the minimum x value for group C?
df[df.y == 'C'].x.min()
df.groupby(df.y)['x'].agg(['mean','min','max']).loc['C']['min']


df.groupby('y').x.sum().idxmax()
# 2. Time Aggregates

# What is the sum of the x values for '2018-05-01'

df.resample('D').sum().x[1]

# What is the average x value for each day? the median?
df.resample('D').agg(['mean', 'median'])

# What day has the largest x value?
df.resample('D').max().x.idxmax()
# which day has the smallest x value for group c?
min_c = df.groupby('y').min().index == 'C'

df[df.x == df.groupby('y').min()[min_c].x['C']].index[0]

# 3. Visualization
# - Visualize the minumum x value of each group with a bar chart.
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns

min_df = df.groupby(df.y).min().reset_index()

plt.bar(min_df.y, min_df.x)

# - Visualize x over time.
df.x.plot()

groups = np.unique(df.y.values)
colors = ['Red', 'Blue', 'Green']

df.groupby('y').plot(legend=True)
plt.show()

df.x[df.y=='A'].plot(color = 'Red', label = 'Group A')
df.x[df.y=='B'].plot(color = 'Blue', label = 'Group B')
df.x[df.y=='C'].plot(color = 'Green', label = 'Group C')
plt.legend()
