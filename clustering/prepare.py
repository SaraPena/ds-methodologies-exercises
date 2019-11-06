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

# Acquire data (for tests)
# df = acquire.get_data_from_mysql()


"""
Remove any properties that are likely to be something other than single unit properties.
(e.g. no duplexes, no land/lot, ...). 
There are multiple ways to estimate that a property is a single unit, and there is not a single "right" answer. But for this exercise, do not purely filter by unitcnt as we did previously. Add some new logic that will reduce the number of properties that are falsely removed.
You might want to use bedrooms, square feet, unit type or the like to then identify those with unitcnt not defined.

"""
def zillow_single_unit(df):
    criteria_1 = df.propertylandusetypeid == 261
    criteria_2 = df.bathroomcnt > 0
    criteria_3 = df.bedroomcnt > 0
    criteria_4 = df.calculatedfinishedsquarefeet > 0
    df = df[(criteria_1) &\
            (criteria_2) &\
            (criteria_3) &\
            (criteria_4)]
    return df

# zillow_single_unit(df)

"""
Create a function that will drop rows or columns based on the percent of values that are missing: handle_missing_values(df, prop_required_column, prop_required_row).
The input:
A dataframe
A number between 0 and 1 that represents the proportion, for each column, of rows with non-missing values required to keep the column. i.e. if prop_required_column = .6, then you are requiring a column to have at least 60% of values not-NA (no more than 40% missing).
A number between 0 and 1 that represents the proportion, for each row, of columns/variables with non-missing values required to keep the row. For example, if prop_required_row = .75, then you are requiring a row to have at least 75% of variables with a non-missing value (no more that 25% missing).
The output:
The dataframe with the columns and rows dropped as indicated. Be sure to drop the columns prior to the rows in your function.
hint:
Look up the dropna documentation.
You will want to compute a threshold from your input values (prop_required) and total number of rows or columns.
Make use of inplace, i.e. inplace=True/False.
"""

def remove_columns(df, cols_to_remove):
    df = df.drop(columns=cols_to_remove)
    return df

def handle_missing_values(df, prop_required_column = .4, prop_required_row = .01):
    threshold = int(round(prop_required_column*len(df.index)))
    df.dropna(axis = 1, thresh = threshold, inplace = True)
    threshold = int(round(prop_required_row*len(df.columns)))
    df.dropna(axis = 0, thresh = threshold, inplace = True)
    return df

def data_prep(df, cols_to_remove=[], prop_required_column=.5, prop_required_row = .75):
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df

# Test
# df = data_prep(df)

def fill_missing_values(df,fill_value):
    df.fillna(fill_value)
    return df
