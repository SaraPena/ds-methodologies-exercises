import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import acquire
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def prep_iris(inverse_transform = False):
    df_iris = acquire.get_iris_data()
    df_iris.drop(columns = ['species_id', 'measurement_id'], inplace = True)
    df_iris.rename(columns = {'species_name': 'species'}, inplace = True)
    encoder = LabelEncoder()
    encoder.fit(df_iris.species)
    df_iris['species'] = encoder.transform(df_iris.species)
    if inverse_transform :
        df_iris.species = pd.Series(encoder.inverse_transform(df_iris.species))
    return df_iris, encoder

# df_iris, encoder = prep_iris()
# df_iris.head()
# encoder

def prep_titanic():
    
    # Acquire titanic dataset
    df_titanic = acquire.get_titanic_data()
    
    # Fill NA values with np.nan
    df_titanic.fillna(np.nan, inplace = True)

    # Drop deck column
    df_titanic.drop(columns = ['deck'], inplace = True)

    # Split dataframe into train, test
    train, test = train_test_split(df_titanic, test_size = .3, random_state = 123, stratify = df_titanic.survived)

    # Train DataFrame: Fill values with 'most_frequent' that are np.NAN in embarked, embark_town
    imp_mode = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    imp_mode.fit(train[['embarked','embark_town']])
    train[['embarked','embark_town']] = imp_mode.transform(train[['embarked','embark_town']])

    # Test DataFrame: Put values with 'most_frequent' are np.NaN in embarked, embark_town
    imp_mode = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    imp_mode.fit(test[['embarked','embark_town']])
    test[['embarked','embark_town']] = imp_mode.transform(test[['embarked', 'embark_town']])

    # Change categorical variables in 'embarked' to numerical values
    int_encoder = LabelEncoder()
    int_encoder.fit(train[['embarked']])
    train[['embarked']] = int_encoder.transform(train[['embarked']])

    int_encoder = LabelEncoder()
    int_encoder.fit(test['embarked'])
    test[['embarked']] = int_encoder.transform(test[['embarked']])

    int_encoder = LabelEncoder()
    int_encoder.fit(train[['embark_town']])
    train[['embark_town']] = int_encoder.transform(train[['embark_town']])

    int_encoder = LabelEncoder()
    int_encoder.fit(test[['embark_town']])
    test[['embark_town']] = int_encoder.transform(test[['embark_town']])

    # Scale age and fare using MinMaxScaler
    scaler = MinMaxScaler()
    train[['age', 'fare']] = scaler.fit_transform(train[['age','fare']])
    test[['age','fare']] = scaler.transform(test[['age','fare']])
    return train, test, int_encoder

# train, test, int_encoder = prep_titanic()

# train.head()
# test.head()
# int_encoder



    