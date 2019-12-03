import numpy as np
import pandas as pd
import math
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def evaluate(actual, predictions,output = True):
    mse = metrics.mean_squared_error(actual, predictions)
    rmse = math.sqrt(mse)

    if output:
        print('MSE: {}'.format(mse))
        print('RMSE: {}'.format(rmse))
    else:
        return mse, rmse

def plot_and_eval(predictions, actual, train, test, metric_fmt = '{:2f}', linewidth=4):
    if type(predictions) is not list:
        predictions = [predictions]
    
    plt.figure(figsize = (16,8))
    plt.plot(train, label = 'Train')
    plt.plot(test, label = 'Test')

    for yhat in predictions:
        mse, rmse = evaluate(actual, yhat, output=False)
        label = f'{yhat.name}'
        if len(predictions) > 1:
            label = f'{label} -- MSE: {metric_fmt} RMSE: {metric_fmt}'.format(mse,rmse)
        plt.plot(yhat,label=label, linewidth=linewidth)

    if len(predictions) == 1:
        label = f'{label} -- MSE: {metric_fmt} RMSE: {metric_fmt}'.format(mse,rmse)
        plt.title(label)
    
    plt.legend(loc = 'best')
    plt.show()





