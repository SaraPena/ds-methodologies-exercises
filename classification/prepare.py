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
    
    # Make the passenger_id the index of the dataset
    df_titanic.set_index('passenger_id', inplace = True)
    
    # df_titanic.head()
    # Look at how many null values are in each column
    # df_titanic.isnull().sum()
    # df_titanic.shape
    
    # Fill null values with np.nan
    df_titanic.embark_town.fillna('Other', inplace = True)
    df_titanic.embarked.fillna('Other', inplace = True)

    # Deck column had 688 null values out of 891 rows. 
    # Because the majority of values are empty we do not not have enough information to go off of. 
    # We will drop 'deck' column because we cannot use the data in this analysis
    df_titanic.drop(columns = ['deck'], inplace = True)

    # Split dataframe into train, test
    train, test = train_test_split(df_titanic, test_size = .3, random_state = 123, stratify = df_titanic.survived)
    

    # Train DataFrame: Fill values with 'most_frequent' that are np.NAN in embarked, embark_town
    imp_mode = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    imp_mode.fit_transform(train[['embarked','embark_town']])
    test[['embarked','embark_town']] = imp_mode.transform(test[['embarked', 'embark_town']])

    # Change categorical variables in 'embarked' to numerical values
    int_encoder = LabelEncoder()
    int_encoder.fit(train[['embarked']])
    train['embarked_encoded'] = int_encoder.transform(train[['embarked']])
    test['embarked_encoded'] = int_encoder.transform(test[['embarked']])

    train.head()


    # Scale age and fare using MinMaxScaler
    scaler = MinMaxScaler()
    train[['age', 'fare']] = scaler.fit_transform(train[['age','fare']])
    test[['age','fare']] = scaler.transform(test[['age','fare']])
    

    return train, test, int_encoder

# train, test, int_encoder = prep_titanic()

# train.head()
# test.head()
# int_encoder



    