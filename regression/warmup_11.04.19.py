import pydataset as data
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
df = data.data('faithful')
info = data.data('faithful', True)
df.info()
df.head()
df.corr()
sns.scatterplot(x='waiting', y='eruptions', data=df)


lm1 = LinearRegression()
lm1.fit(df[['waiting']], df[['eruptions']])

y_pred_lm1 = lm1.predict(df[['waiting']])

r2_lm1 = r2_score(df[['eruptions']],y_pred_lm1)


y_pred1 = pd.DataFrame(y_pred_lm1, columns = df[['eruptions']].columns.values).set_index([df.index.values])
y_pred1 = y_pred1['eruptions']

model = pd.DataFrame({'waiting': df['waiting'],
                    'eruptions':df['eruptions'],
                    'lm1': y_pred1.ravel()})
sns.set_style('whitegrid')
sns.scatterplot(x = 'waiting', y = 'eruptions', data = model, color = 'blue', label = 'actual' )
sns.scatterplot(x = 'waiting', y = 'lm1', data = model, color = 'orange', label = 'predictions')
plt.ylabel('eruptions')
plt.title('predictions vs. actual')
plt.legend()