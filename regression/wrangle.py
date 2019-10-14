# Throughout the exercises for Regression in Python lessons, you will use the following example scenario: 
#   As a customer analyst, I want to know who has spent the most money with us over their lifetime. 
#   I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. 
#   I need to do this within an average of $5.00 per customer.

# The first step will be to acquire and prep the data. Do your work for this exercise in a file named wrangle.py.

# Import needed libraries to clean data : warnings, pandas(DataFrames), matplotlib.pyplot, seaborn,numpy,env
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from env import host, user, password


# Define function to create url address for sql databases

def get_db_url(db_name):
    """Form db url using database name """
    url = f'mysql+pymysql://{user}:{password}@{host}/{db_name}'
    return url

url = get_db_url('telco_churn')

# Create query for data to aquire from SQL

# query = """SELECT customer_id, monthly_charges, tenure, total_charges
            #FROM customers
            #WHERE contract_type_id = 3"""

def read_sql_file():
    """use pd.read_sql to create your dataframe (df).
    \nParameters: query (SQL format), url (name directly or use get_db_url()) """
    query = """SELECT * 
               FROM customers as c 
               JOIN internet_service_types as i USING (internet_service_type_id)
               WHERE contract_type_id = 3 """
    
    df = pd.read_sql(query, url)
    return df

tc = read_sql_file()


tc = pd.read_sql("SELECT * FROM customers as c JOIN internet_service_types as i on c.internet_service_type_id = i.internet_service_type_id",url)
# Gather infomation about data
# tc.head()
# tc.shape
# tc.describe()
# tc.info()

# tc.isnull().sum()

# tc.columns[tc.isnull().any()]

# tc.total_charges.value_counts(sort=True, ascending=True)

# tc.replace(r'^\s*$', np.nan, regex=True, inplace=True)
# tc.info()

# tc = tc.dropna()

# tc['total_charges'] = pd.to_numeric(tc.total_charges, errors = 'coerce').astype('float')

def wrangle_telco():
    """ Handle dtypes, and empty values in dataframe (df) """
    get_db_url('telco_churn')
    df = read_sql_file()
    df.replace(r'^\s*$', np.nan, regex = True, inplace=True)
    df['total_charges'] = pd.to_numeric(df.total_charges, errors = 'coerce').astype('float')
    df = df.dropna()
    df = df.drop(columns = 'internet_service_type_id').set_index('customer_id')
    return df

#wrangle_telco().info()


