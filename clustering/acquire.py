import pandas as pd
import numpy as np

# env.py - contains variables user, password, and host to connect to sql database
import env

def get_db_url(db):
    """ function uses your env file for variables: env.user, env.password, env.host
        \n db: database name, exambles: 'zillow', 'telco' """
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

# Example:
# url = get_db_url('zillow')

def get_data_from_mysql():
    """use pd.read_sql to get data from query variable that is a syntatically correct sql query, and db tha is an existing database on your sql host
        \n returns dataframe of query """
        
    query = """SELECT logerror, transactiondate, p.*
    FROM predictions_2017 p_17
    JOIN 
        (SELECT
        parcelid, Max(transactiondate) as tdate
        FROM predictions_2017
        GROUP BY parcelid )as sq1
    		on (sq1.parcelid=p_17.parcelid and sq1.tdate = p_17.transactiondate)
    JOIN properties_2017 p on p_17.parcelid=p.parcelid
    WHERE p.latitude IS NOT NULL and p.longitude IS NOT NULL and YEAR(p_17.transactiondate) = '2017';"""

    db = 'zillow'
    
    df = pd.read_sql(query, get_db_url(db))

    return df
    
# Example:
# df = get_data_from_mysql()
# df.head()









