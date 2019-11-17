import warnings
warnings.filterwarnings('ignore')

import json
from pprint import pprint

import matplotlib.pyplot as plt 
import seaborn as sns

import pandas as pd 

from os import path 

import acquire

def prepare_store_data():
    # convert date to date_time format
    df = acquire.get_all_data()
    df['sale_date'] = pd.to_datetime(df['sale_date'])

    #sort df by `sale_date`
    df.sort_values('sale_date', inplace = True)

    #set the index to tbe the datetime variable
    by_date = df.set_index('sale_date')

    # create 'month' column
    by_date['month'] = list(by_date.index.month)

    # create 'nameofdayofweek' column
    by_date['nameofdayofweek'] = list(by_date.index.weekday_name)

    # create 'dayofweek'
    by_date['dayofweek'] = list(by_date.index.dayofweek)

    # create 'sales_total' column
    by_date['sales_total'] = by_date['sale_amount']*by_date['item_price']

    return by_date

 df = prepare_store_data()

def sales_by_day(df):
    sales_by_day = df[['sales_total']].resample('D').sum()
    sales_by_day['daily_sales_diff'] = sales_by_day.sales_total.diff()
    return sales_by_day

# sales_by_day(df)