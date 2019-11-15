import requests
import warnings
warnings.filterwarnings('ignore')
import json
from pprint import pprint

import pandas as import pd

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

def get_store_data_from_api():
    







