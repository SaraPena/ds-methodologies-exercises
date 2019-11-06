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
import prepare

def wrangle_zillow_data():
    df = acquire.get_zillow_data()
    df = prepare.zillow_single_unit(df)
    df = prepare.remove_columns(df,['finishedsquarefeet12','fullbathcnt', 'unitcnt','structuretaxvaluedollarcnt','landtaxvaluedollarcnt', 'assessmentyear', 'propertyzoningdesc'])
    df = prepare.handle_missing_values(df)
    return df

# wrangle_zillow_data().info()
# wrangle_zillow_data().fips.value_counts()


