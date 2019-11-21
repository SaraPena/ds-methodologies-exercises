import requests
import warnings
warnings.filterwarnings('ignore')
import json
from pprint import pprint


from os import path

import pandas as pd

def get_item_data_from_api():

    base_url = 'https://python.zach.lol'

    response = requests.get(base_url)
    data = response.json()
    api = data['api']
    data['api']

    api_base = base_url + api

    response = requests.get(api_base)
    data = response.json()
    routes = data['payload']['routes']
    items_hp = routes[routes.index('/items')]

    items_data = requests.get(base_url + api + items_hp).json()
    max_page = items_data['payload']['max_page']

    for i in range(1, max_page+1):
        if i == 1:
            response = requests.get(base_url+api+items_hp)
            data = response.json()
            items = pd.DataFrame(data['payload']['items'])
            print('Fetching page {} of {}'.format(data['payload']['page'], max_page))

            #print('Fetching page {} of {}'.format(data['payload']['page']+1, max_page))
        elif i == max_page:
            items = pd.concat([items, pd.DataFrame(data['payload']['items'])]).reset_index().drop(columns = ['index'])
            print('Fetching page {} of {}'.format(data['payload']['page'], max_page))
            break
        else:
            items = pd.concat([items, pd.DataFrame(data['payload']['items'])])
            print('Fetching page {} of {}'.format(data['payload']['page'], max_page))

        response = requests.get(base_url + data['payload']['next_page'])
        data = response.json()

    return items

# test
# get_item_data_from_api()
    
def get_store_data_from_api():
    base_url = 'https://python.zach.lol'
    response = requests.get(base_url)
    data = response.json()
    api = data['api']

    api_base = base_url + api

    response = requests.get(api_base)
    data = response.json()
    routes = data['payload']['routes']
    stores_hp = routes[routes.index('/stores')]

    stores_data = requests.get(api_base+stores_hp).json()
    max_page = stores_data['payload']['max_page']

    for i in range(1,max_page+1):

        if i == 1 :
            response = requests.get(base_url+api+stores_hp)
            data = response.json()
            stores = pd.DataFrame(data['payload']['stores'])

            print('Fetching page {} of {}'.format(data['payload']['page'], max_page))

        elif i == max_page & i != 1:
            stores = pd.concat([stores, pd.DataFrame(data['payload']['stores'])]).reset_index().drop(columns = ['index'])
            print('Fetching page {} of {}'.format(data['payload']['page'], max_page))
            break
        else:
            stores = pd.concat([stores, pd.DataFrame(data['payload']['stores'])])
            response = requests.get(base_url + data['payload']['next_page'])
            data = response.json()
            print('Fetching page {} of {}'.format(data['payload']['page']+1, max_page))


    return stores
        
# Test
# get_store_data_from_api()

def get_sales_data_from_api():
    base_url = 'https://python.zach.lol'
    response = requests.get(base_url)
    data = response.json()
    base_api = data['api']

    response = requests.get(base_url+base_api)
    data = response.json()
    routes = data['payload']['routes']
    sales_hp = routes[routes.index('/sales')]
    
    sales_data = requests.get(base_url+base_api+sales_hp).json()
    max_page = sales_data['payload']['max_page']

    for i in range(1,max_page +1):
        if i == 1:
            response = requests.get(base_url+base_api+sales_hp)
            data = response.json()
            sales = pd.DataFrame(data['payload']['sales'])

            print('Fetching page {} of {}'.format(data['payload']['page'], max_page))

        elif i == max_page and i !=1:
            sales = pd.concat([sales, pd.DataFrame(data['payload']['sales'])]).reset_index().drop(columns = ['index'])
            print('Fetching page {} of {}'.format(data['payload']['page'], max_page))
            break
        
        else:
            sales = pd.concat([sales, pd.DataFrame(data['payload']['sales'])])
            print('Fetching page {} of {}'.format(data['payload']['page'], max_page))

        response = requests.get(base_url+data['payload']['next_page'])
        data = response.json()

    return sales
# Test
# get_sales_data_from_api()

def get_item_data(use_cache = True):
    if use_cache and path.exists('items.csv'):
        return pd.read_csv('items.csv')
    df = get_item_data_from_api()
    df.to_csv('items.csv', index = False)
    return df

# Test
# get_item_data()

def get_store_data(use_cache= True):
    if use_cache and path.exists('stores.csv'):
        return pd.read_csv('stores.csv')
    df = get_store_data_from_api()
    df.to_csv('stores.csv', index = False)
    return df

# Test
get_store_data()

def get_sales_data(use_cache=True):
    if use_cache and path.exists('sales.csv'):
        return pd.read_csv('sales.csv')
    df = get_sales_data_from_api()
    df.to_csv('sales.csv', index = False)
    return df

# get_sales_data()

def get_all_data():
    sales = get_sales_data()
    stores = get_store_data()
    items = get_item_data()

    sales = sales.rename(columns = {'item': 'item_id', 'store': 'store_id'})

    return sales.merge(items, on='item_id').merge(stores, on = 'store_id')

# Test
# get_all_data().info()

def get_power_data(use_cache = True):
    if use_cache and path.exists('power.csv'):
        return pd.read_csv('power.csv')
    base_url = 'https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv'
    power = pd.read_csv(base_url)
    power.to_csv('power.csv', index = False)
    return power

# get_power_data()

