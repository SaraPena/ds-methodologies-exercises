import requests
import warnings
warnings.filterwarnings('ignore')
import json
from pprint import pprint

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

            print('Fetching page {} of {}'.format(data['payload']['page']+1, max_page))
        elif i == max_page:
            items = pd.concat([items, pd.DataFrame(data['payload']['items'])]).reset_index().drop(columns = ['index'])
            print('Fetching page {} of {}'.format(data['payload']['page'], max_page))
            break
        else:
            items = pd.concat([items, pd.DataFrame(data['payload']['items'])])
        response = requests.get(base_url + data['payload']['next_page'])
        data = response.json()

    return items

# test
# get_item_data_from_api()
    
def get_item_data_from_api():
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





