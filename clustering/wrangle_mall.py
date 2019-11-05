# 1. Acquire data from mall_customers.customers in mysql

# 2. Split the data

# 3. One hot encoding

# 4. Missing Values - None in mall data.

# 5. Scaling
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

import acquire
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

def wrangle_mall():
    df = acquire.get_mallcustomer_data()

    train, test =  train_test_split(df, random_state = 42)

    int_encoder = LabelEncoder()
    int_encoder.fit(train.gender)
    train['gender_bool'] = int_encoder.transform(train.gender)
    test['gender_bool'] = int_encoder.transform(test.gender)

    ohe = OneHotEncoder(sparse = False, categories = 'auto')
    gender_ohe = ohe.fit_transform(train[['gender_bool']])
    gender_ohe = pd.DataFrame(gender_ohe, columns = ['Female', 'Male'], index = train.index)
    train.drop(columns = ['gender_bool'], inplace = True)
    train = train.join(gender_ohe)

    scaler = MinMaxScaler()
    num_vars = list(train.select_dtypes('number').columns)
    train[num_vars] = scaler.fit_transform(train[num_vars])
    return df, train, test

# df, train, test = wrangle_mall()
# train.head()