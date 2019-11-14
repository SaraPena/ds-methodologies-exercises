# Exercises
# For all of the datasets below, examine the data types of each column, ensure that the dates are in the proper format, and set the dataframe's index to the date column as appropriate.

# For this exercise you'll need to install a library that will provide us access to some more datasets:

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from vega_datasets import data

data.sf_temps()

# Use the above dataset for the exercises below:
df = data.sf_temps()
df.info()
df.set_index('date',inplace = True)
df.sort_index(inplace=True)

# 1. Resample by the day and take the average temperature. Visualize the average temperature over time.
by_day_mean = df.resample('D').mean()

by_day_mean.plot()

# 2. Write the code necessary to visualize the minimum temperature over time.
by_day_min = df.resample('D').min()

by_day_min.plot()
by_day_mean.plot()
plt.show()

plt.scatter(x = list(by_day_min.index), y = by_day_min['temp'])
plt.scatter(x = list(by_day_mean.index), y = by_day_mean['temp'])

#3. Write the code necessary to visualize the maximum temperature over time.
by_day_max = df.resample('D').max()

by_day_max.plot()

plt.plot(by_day_min, label = 'min', color = 'blue')
plt.plot(by_day_mean, label = 'mean', color = 'black')
plt.plot(by_day_max, label = 'max', color = 'red')
plt.legend()
plt.title('Temperature stats')

#4 Which month is the coldest, on average?
cold_months = df.resample('D').min().resample('M').mean()
cold_months[cold_months.temp == min(cold_months.temp)]
min(cold_months.temp)

#5 Which month has the highest average temperature?
average_temps = df.resample('M').mean()
average_temps[average_temps.temp == max(average_temps.temp)]

#6 Resample by the day and calculate the min and max temp for the day (Hint: .agg(['min','max'])).
#  Use the resampled dataframe to calculate the change in temperature for the day. Which month has the highest daily temperature variability?

min_max = df.resample('D').agg(['min','max'])
min_max['temp_diff'] = min_max['temp']['max'] - min_max['temp']['min']
temp_var = min_max[['temp_diff']].resample('M').mean()
temp_var[temp_var.temp_diff == max(temp_var.temp_diff)]

# Use the dataset to answer the following questions:
from vega_datasets import data

# Which year and month combination has the highest amount of precipitation
df = data.seattle_weather().set_index('date')
rain = df.resample('M').sum()[['precipitation']]
rain[rain.precipitation == max(rain.precipitation)]

yearly_rain = df.resample('Y').sum()[['precipitation']]
yearly_rain[yearly_rain.precipitation == max(yearly_rain.precipitation)].reset_index()['date'].dt.year[0]

# Visualize the amount of monthly precipitation over time.
df.resample('M').sum()['precipitation'].plot(color='blue')

# Visualize the amount of wind over time. Choose a time interval you think is appropriate?
df.resample('M').mean()['wind'].plot(color = 'black')



