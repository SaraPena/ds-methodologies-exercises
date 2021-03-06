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

# Which year and month combination has the highest amount of precipitation
df = data.seattle_weather().set_index('date')
rain = df.resample('M').sum()[['precipitation']]
rain[rain.precipitation == max(rain.precipitation)]

yearly_rain = df.resample('Y').sum()[['precipitation']]
yearly_rain[yearly_rain.precipitation == max(yearly_rain.precipitation)].reset_index()['date'].dt.year[0]

# Visualize the amount of monthly precipitation over time.
df.resample('M').sum()['precipitation'].plot(color='blue')

# Visualize the amount of wind over time. Choose a time interval you think is appropriate?
df.resample('Q').mean()['wind'].plot(color = 'black')

wind_months = df.resample('D').max().resample('M').mean()['wind']

wind_months[wind_months == max(wind_months)]

wind_mean = df.resample('D').mean().resample('M').mean()['wind']
wind_mean[wind_mean == max(wind_mean)]

# What's the sunniest year? (Hint: which day has the highest number of days where weather == sun)
sun_yearly = df[['weather']][df.weather == 'sun'].resample('Y').count()
sun_yearly[sun_yearly.weather == max(sun_yearly.weather)].reset_index()['date'].dt.year[0]
list(sun_yearly[sun_yearly.weather == max(sun_yearly.weather)].index)[0].year

# In which month does it rain the most?
df.weather.value_counts()
rain_months = df[['weather']][df.weather == 'rain'].resample('M').count()
list(rain_months[rain_months.weather == max(rain_months.weather)].index)[0].month

# which month has the most number of days with a nonzero amount of precipitation?
preciptiation_days_monthly = df[df.precipitation>0]['precipitation'].resample('M').count()
preciptiation_days_monthly[preciptiation_days_monthly==max(preciptiation_days_monthly)]

### flights_20k

df = data.flights_20k()
df.info()
df.delay.value_counts().sort_index()[0]

df['delay'][df['delay']<0] = 0

df['delay'].value_counts().sort_index()

df.set_index('date', inplace=True)

delay_hours = df.groupby(df.index.hour).mean()['delay']
delay_hours[delay_hours == max(delay_hours)].rename_axis('minutes')

delay_day = df.groupby(df.index.weekday_name).mean()['delay']
delay_day[delay_day == max(delay_day)]

delay_month = df.groupby(df.index.month).mean()['delay']

df.index.month.value_counts()
