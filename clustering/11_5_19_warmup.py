import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('50k-posts-from-relationship-advice.csv', header = None)
df.rename(columns = {0: 'Female', 1:'Male', 2: 'id', 3:'Posts'}, inplace = True)
df.head()
df.info()
df.shape

df[['Female']].hist(bins = 10)
plt.xlim((0,100))

df.Female.value_counts(bins = 10)

drop_id = list(df[(df.Female > 90) | (df.Female < 13)].index)

df.drop(index = drop_id, inplace = True)

df.Female.value_counts(bins = 10).sort_index()

df.Female.describe()

df.Male.describe()
df.Male.value_counts(bins = 10).sort_index()

drop_id = list(df[df.Male > 90])