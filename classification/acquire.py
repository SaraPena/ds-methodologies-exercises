import pandas as pd
import numpy as np

import env

def get_db_url(db):
    """ function uses your env file for variables:  env.user, env.password, env.host
    \n db: database name, examples: 'zillow', 'telco' """
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def get_data_from_mysql(query, db):
    """ use pd.read_sql to get data from query variable that is a syntatically correct sql query, and db that is an existing database on your sql host.
    \n returns dataframe of query, and url with pd.read_sql """
    df = pd.read_sql(query, get_db_url(db))
    return df

def clean_data(df):
    df = df.dropna()
    return df
  
def wrangle_zillow():
    df = get_data_from_mysql()
    df = clean_data(df)
    return df

#def wrangle_telco():
   # return clean_data(get_data_from_mysql())

def get_iris_data():
    return sns.load_dataset('iris')

def get_custdetails_data():
    return pd.read_excel('Excel_Exercises.xlsx', sheet_name =0)

def get_train_data():
    sheet_url = 'https://docs.google.com/spreadsheets/d/1Uhtml8KY19LILuZsrDtlsHHDC9wuDGUSe8LTEwvdI5g/edit#gid=341089357'
    csv_export_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
    return pd.read_csv(csv_export_url)
    
def get_titanic_data(db):
    return pd.read_sql('SELECT * FROM passengers', get_db_url(db))
