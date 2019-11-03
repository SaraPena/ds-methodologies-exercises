# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Wrangling
import pandas as pd 
import numpy as np 

# Exploring
import scipy.stats as stats
import matplotlib.pyplot as plt 
import seaborn as sns 

# default pandas decimal number display format.
pd.options.display.float_format = '{:20,.2f}'.format

import acquire

df = acquire.get_data_from_mysql()

def nulls_by_col(df):
    # Look at the number missing.
    num_missing = df.isnull().sum()

    # number of rows
    rows = df.shape[0]

    # percent_missing
    pct_missing = num_missing/rows

    # Create dataframe of column name, num_rows missing, and pct of rows missing.
    cols_missing = pd.DataFrame({'num_rows': num_missing, 'pct_rows_missing': pct_missing})

    return cols_missing

# Test:
# nulls_by_col(df)

def nulls_by_row(df):
    # look at nulls by rows (axis = 1)
    num_cols_missing = df.isnull().sum(axis = 1)
    num_cols_missing.value_counts()

    # number of columns
    columns = df.shape[1]

    # Percents of columns missing in each row:
    pct_cols_missing = num_cols_missing/columns * 100

    # Create a data frame that shows how many rows are missing x amount of columns and the pcts of columns.
    rows_missing = pd.DataFrame({'num_cols_missing': num_cols_missing, 
                                 'pct_cols_missing': pct_cols_missing})\
                                 .reset_index()\
                                 .groupby(['num_cols_missing', 'pct_cols_missing'])\
                                 .count()\
                                 .rename(index = str, columns = {'index': 'num_rows'})\
                                 .reset_index()
    
    return rows_missing

#Test
# nulls_by_row(df)








