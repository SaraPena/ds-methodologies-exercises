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

df = acquire.get_zillow_data()

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

def df_value_counts(df):
    for col in df.columns:
        print(f'{col}:')
        if df[col].dtype == 'object':
            col_count = df[col].value_counts()
        else:
            if df[col].nunique() >= 35:
                col_count = df[col].value_counts(bins=10, sort = False)
            else:
                col_count = df[col].value_counts()
        print(col_count)
        print('\n')

# Test
# df_value_counts(df)

def get_outliers(s, k):
    """
    Given a series and a cutoff value, k, returns the upper outliers for the series.
    
    The values returned will be either 0 (if the point is not an outlier), or a number that indicated how far away from the upper bound the observation is.
    """
    
    q1, q3 = s.quantile([.25,.75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    lower_bound = q1 - k * iqr
    return s.apply(lambda x: max([x - upper_bound,0])), s.apply(lambda x: min([x - lower_bound,0]))

def add_outlier_columns(df,k):
    """
    Add a column with the suffix _outliers for all the numeric
    """
    for col in df.select_dtypes('number'):
        df[col + '_lower_outliers'] = get_outliers(df[col],k)[1]
        df[col + '_upper_outliers'] = get_outliers(df[col],k)[0]
    return df

def df_summary(df):
    print(f'--- Shape:{df.shape}')
    print('\n--- Info :')
    df.info()
    print('\n--- Descriptions:')
    print(df.describe(include='all'))
    print(f'\n--- Nulls by Column: \n {nulls_by_col(df)}')
    print('\n--- Value Counts:\n')
    print(df_value_counts(df))

# Test
# df_summary(df)













