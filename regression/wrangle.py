# Throughout the exercises for Regression in Python lessons, you will use the following example scenario: 
#   As a customer analyst, I want to know who has spent the most money with us over their lifetime. 
#   I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. 
#   I need to do this within an average of $5.00 per customer.

# The first step will be to acquire and prep the data. Do your work for this exercise in a file named wrangle.py.

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

