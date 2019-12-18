import pandas as pd 
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from pydataset import data

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# 1. Use `pydataset` to load the `voteincome` dataset

data('voteincome', show_doc=True)

df = data('voteincome')
df.info()

df.vote.value_counts()


# 2. Drop the `state` and `year` columns.
df.drop(columns = ['state', 'year'], inplace = True)

y = df[['vote']]
X = df.drop(columns = ['vote'])
X.info()


# 3. split the data into train, and test datasets. We will be predicting whether or not someone votes based on the remaining features.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123)



# 4. Fit a k-neighbors classifer on the training data. Use 4 for your number of neighbors.
#    How accurate is your model?
#    How does it perform on the test data?
#Create KNN Object
knn = KNeighborsClassifier(n_neighbors = 4)

# Fit the model to the training data
knn.fit(X_train, y_train)

# Estimate whether or not a person will vote, using the training data.

y_pred = knn.predict(X_train)

best = [0]
for k in range(1,5):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    print(f'for k = {k}')
    print('Accuracy of KNN classifer on test set: {:.2f}'.format(knn.score(X_test, y_test)))
    if knn.score(X_test, y_test) > best[0]:
        best = [knn.score(X_test,y_test)]
        ypred = knn.predict(X_test)




# 6. View the classification report for your best model.
print(classification_report(y_test,ypred))

import matplotlib.pyplot as plt
k_range = range(1, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])


max(scores)

x = pd.Series(scores)
x[x == max(x)]

best = [0]
for k in range(1,20):
    knn = KNeighborsClassifier(n_neighbors = k, weights = 'uniform')
    knn.fit(X_train, y_train)
    print(f'for k = {k}')
    print('Accuracy of KNN classifer on test set: {:.2f}'.format(knn.score(X_test, y_test)))
    if knn.score(X_test, y_test) > best[0]:
        best = [knn.score(X_test,y_test)]
        ypred = knn.predict(X_test)

print(classification_report(y_test,ypred))


from sklearn import __version__

__version__

