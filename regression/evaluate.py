import pandas as pd 
import numpy as np 
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression
from math import sqrt
import matplotlib.pyplot as plt 
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# 1. Load the tips dataset from either pydataset or seaborn.

df = sns.load_dataset('tips')
df.describe()
df.info()
df.corr()
df.size
df.shape

# 2. Fit a linear regression model (ordinary least squares) and compute yhat, predictions of tip using total_bill.
#    You may follow these steps to do that:

#    import the method from statsmoduls: from statsmodels.formula.api import ols
#    - DONE ABOVE

#    Fit the model to your data where x = total_bill and y = tip: regr = ols('y-x', data=df).fit


tip = df['tip']
total_bill = df['total_bill']

ols_model = ols('tip~total_bill',data=df).fit()

# compute yhat, the predictions of tip using total_bill: df['yhat'] = ols_model.predict(df.total_bill)

df['yhat'] = ols_model.predict(total_bill)

df[['tip','yhat']]

ols_model.summary()

# 3. Write a function, plot_residuals(x, y, dataframe) that takes the feature, the target, and the dataframe as input and returns the residual plot(hint:seaborn has an easy way to do this!)

def plot_residuals(x,y,dataframe):
    return sns.residplot(x, y,data = dataframe, color = 'Green')

sns.set_style('whitegrid')
plot_residuals('total_bill', 'tip', df)

# 4. Write a function, regression_errors(y, yhat), that takes in y and yhat, returns the sum of squared errors(SSE), explained sum of squares(ESS), total sum of squares(TSS), mean squared error(MSE), and root mean squared error (RMSE).
def regression_errors(y, yhat):
    sse = mean_squared_error(y, yhat)*len(y)
    ess = sum((yhat - y.mean())**2)
    tss = sse + ess
    mse = mean_squared_error(y, yhat)
    rmse = sqrt(mean_squared_error(y, yhat))
    df_eval = pd.DataFrame(np.array(['SSE', 'ESS','TSS','MSE','RMSE']), columns = ['metric'])
    df_eval['model_error'] = np.array([sse, ess, tss, mse, rmse])
    return df_eval

df_eval = regression_errors(df.tip,df.yhat)

# 5. Write a function, baseline_mean_errors(y), that takes in your target,y, computes SSE, MSE, & RMSE when yhat is equal to the mean of all y, and returns the error values (SSE, MSE, and RMSE)

def baseline_mean_errors(y):
    df_baseline = pd.DataFrame(y)
    df_baseline['y.mean()'] = y.mean()
    df_baseline['residual'] = df_baseline['y.mean()'] - y
    df_baseline['residual^2'] = df_baseline['residual'] **2
    sse = sum(df_baseline['residual^2'])
    mse = sse/len(y)
    rmse = sqrt(mse)
    df_baseline_error = pd.DataFrame({'metric': ['SSE', 'MSE', 'RMSE'], 'baseline_error': [sse, mse, rmse]})
    return df_baseline_error

df_baseline_error = baseline_mean_errors(df.tip)

# 6. Write a function, better_than_baseline(SSE), that returns true if your model performs better than the baseline, otherwise false

(df_eval[df_eval.metric == 'SSE']).model_error.sum()
(df_baseline_error[df_baseline_error.metric == 'SSE']).baseline_error.sum()

# Create a function to predit yhat values

def yhat_predict(x,y,df):
    y = pd.DataFrame(y)
    x = pd.DataFrame(x)
    ols_model = ols('y~x',data=df).fit()

    # compute yhat, the predictions of tip using total_bill: df['yhat'] = ols_model.predict(df.total_bill)

    df['yhat'] = ols_model.predict(x)
    return df['yhat']

yhat(df.tip,df.total_bill,df)


def better_than_baseline(x,y,df):
    yhat = yhat_predict(x,y,df)
    df_eval = regression_errors(y,yhat)
    df_baseline_error = baseline_mean_errors(y)
    sse_model = (df_eval[df_eval.metric == 'SSE']).model_error.sum()
    sse_baseline = (df_baseline_error[df_baseline_error.metric == 'SSE']).baseline_error.sum()
    return sse_model < sse_baseline

better_than_baseline(df.total_bill, df.tip, df)

# 8. Write a function, model_significance(ols_model), that takes the ols model as input and returns the amount of variance explained in your model, and the value telling you whether the correlation between the model and the tip value are statistically significant.

tip = df['tip']
total_bill = df['total_bill']

ols_model = ols('tip~total_bill',data=df).fit()

def model_significance(ols_model):
    evs = ols_model.rsquared
    f_pval = ols_model.f_pvalue
    return f'evs: {evs:.4f}, f_pval: {f_pval}'

model_significance(ols_model)
