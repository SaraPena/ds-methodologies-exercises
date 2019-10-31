# Excercises

# Do you work for these excercises in either a notebook or a python script named model.

# 1. Fit the logistic regression classifier to your training sample and transform, i.e. make preictions on the training sample.

import numpy as np
import pandas as pd

from pydataset import data

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

import graphviz
from graphviz import Graph

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from acquire import get_iris_data
from acquire import get_titanic_data
from prepare import prep_iris
from prepare import prep_titanic


# Create iris data set, and get encoder to transform data.
df, encoder = prep_iris(inverse_transform=True)

df.head()
df.dtypes

# In our prep we know that the dataset is set up how we want by droping species_id and renaming to species. Dropping measurement_id.

train, test = train_test_split(df, test_size = .3, random_state = 123, stratify = df[['species']])

X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = train[['species']]

train.describe()
train.info()

# From sklearn.linear_model import LogisticRegression
# Create the model object
logit = LogisticRegression(C=1, random_state = 123, solver = 'saga')

# Try with not setting parameters
lr = LogisticRegression()

# Fit the model:
logit.fit(X_train, y_train)
lr.fit(X_train, y_train)
pred_y_lr = lr.predict(X_train)



# Look at Coeeficients:
logit.coef_

# Look at the intercepts:
logit.intercept_

# Make predictions and look at the probabilities
y_pred = logit.predict(X_train)
y_pred_proba = logit.predict_proba(X_train)

# Compute the accuracy of the model with logit.score
logit.score(X_train,y_train)
lr.score(X_train, y_train)
# Evaluate your in-sample results using the model score, confusion matrix, and classification report.

# Create a confusion matrix using sklearn.metrics confusion_matrix function
confusion_matrix(y_train, y_pred)

# Print the classification_report using sklearn.metrics
print(classification_report(y_train,y_pred))

# Create your confusion matrix variables for each species of flowers: sertosa, versicolor, virginica.
# look at <https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal> for how to read a confusion matrix with more than two classes as outcomes.
# From research we have found that the confusion matrix alphabetically orders the variables on the rows, and columns of the confustion matrix.

cmat = confusion_matrix(y_train, y_pred)

sertosa_true_positive = cmat[0,0]
sertosa_true_negative = cmat[1,1] + cmat[2,2]
sertosa_false_negative = cmat[0,1] + cmat[0,2]
sertosa_false_positive = cmat[1,0] + cmat[2,0]

versicolor_true_positive = cmat[1,1]
versicolor_true_negative = cmat[0,0] + cmat[0,2] + cmat[2,0] + cmat[2,2]
versicolor_false_negative = cmat[1,0] + cmat[1,2]
versicolor_false_positive = cmat[0,1] + cmat[2,1]

virginica_true_positive = cmat[2,2]
virginica_true_negative = cmat[1,1] + cmat[0,0]
virginica_false_negative = cmat[2,0] + cmat[2,1]
virginica_false_positive  = cmat[0,2] + cmat[1,2]

# Compare by hand formula for accurracy to score result.
accuracy = (((sertosa_true_negative + 
              sertosa_true_positive + 

              versicolor_true_positive + 
              versicolor_true_negative +

              virginica_true_negative +
              virginica_true_positive)/
              (sertosa_false_positive +
               sertosa_false_negative +
               sertosa_true_positive +
               sertosa_true_negative +

               versicolor_false_positive +
               versicolor_false_negative +
               versicolor_true_positive +
               versicolor_true_negative +

               virginica_false_positive +
               virginica_false_negative +
               virginica_true_positive +
               virginica_true_negative)))


classification_error = (((sertosa_false_negative +
                         sertosa_false_positive +
                         
                         versicolor_false_negative +
                         versicolor_false_positive +
                         
                         virginica_false_negative +
                         virginica_false_positive)/
                         (sertosa_false_negative +
                          sertosa_false_positive +
                          sertosa_true_positive +
                          sertosa_true_negative +
                          
                          versicolor_false_positive +
                          versicolor_false_negative +
                          versicolor_true_positive +
                          versicolor_true_negative +
                          
                          virginica_false_negative +
                          virginica_false_positive +
                          virginica_true_negative +
                          virginica_true_positive)))

recall = (((sertosa_true_positive +
            versicolor_true_positive +
            virginica_true_positive)/
            (sertosa_true_positive +
            sertosa_false_negative +
            
            versicolor_true_positive +
            versicolor_false_negative +
            
            virginica_true_positive +
            virginica_false_negative)))

precision = (((sertosa_true_positive +
               versicolor_true_positive +
               virginica_true_positive)/
              (sertosa_true_positive +
               sertosa_false_positive +
               
               versicolor_true_positive +
               versicolor_false_positive +
               
               virginica_true_positive +
               virginica_false_positive)))

f1_score = (recall + precision)/2

false_positive_rate = (((sertosa_false_positive +
                         versicolor_false_positive +
                         virginica_false_positive)/
                        (sertosa_true_negative +
                         sertosa_false_positive +
                         
                         versicolor_true_negative +
                         versicolor_false_positive +
                         
                         virginica_true_negative +
                         virginica_false_positive)))


cr = (classification_report(y_train,y_pred, output_dict = True))

y_pred_proba_setosa = [i[0] for i in y_pred_proba]
y_pred_proba_versicolor = [i[1] for i in y_pred_proba]
y_pred_proba_virginica = [i[2] for i in y_pred_proba]


sns.set_style('whitegrid')

sns.scatterplot(y_pred_proba_setosa,y_pred, color = 'red')
sns.scatterplot(y_pred_proba_versicolor, y_pred, color = 'green')
sns.scatterplot(y_pred_proba_virginica, y_pred, color = 'blue')

y_pred
y_pred_proba


## Decision Tree:

# 1. Fit the decision tree classifer to your training sample and transform (i.e. make predictions on the training sample)

df = data('iris')
df.head()

# clean up the columns:
df.columns = [col.lower().replace('.','_') for col in df]

# Now we'll do our training/test split:

X = df.drop(['species'], axis = 1)
y = df[['species']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123)

X_train.head()

# Train Model

# Create decision tree object:

# for classification you can change the algorithm to gini or entropy (information gain). Default is gini.
# 4. Run through steps 2-4 using 'gini'  as your measure of impurity
clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state = 123)
clf_gini = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state = 123)
# Fit the model to the training data
clf.fit(X_train,y_train)
clf_gini.fit(X_train,y_train)

# Estimate species
y_pred = clf.predict(X_train)
y_pred[0:5]

y_pred_gini = clf_gini.predict(X_train)
y_pred_gini[0:5]

# Estimate the probability of a species
y_pred_proba = clf.predict_proba(X_train)
y_pred_proba

y_pred_proba_gini = clf_gini.predict_proba(X_train)
y_pred_proba_gini


# 2. Evaluate your in-sample results using the model score, confusion_matrix, and classification report.
# Compute the accuracy

accuracy = clf.score(X_train,y_train)
accuracy_gini = clf_gini.score(X_train, y_train)

# Create a confusion matrix, going to store it as dataframe object

# Look at the labels for the target variables:
sorted(y_train.species.unique())

# Look at the train dataframe target variable value counts
y_train.species.value_counts()

# Create the column and row names for confusion matrix dataframe using the unique names we found in y_train.species.unique()
# Sort these values, to create an alpha list. This will align with how the confusion matrix is created with sklearn.metrics
labels = sorted(y_train.species.unique())

# Create the DataFrame for the confusion matrix, and assign it to the variable cmat
cmat = pd.DataFrame(confusion_matrix(y_train,y_pred),index = labels, columns = labels)
cmat_gini = pd.DataFrame(confusion_matrix(y_train, y_pred_gini), index = labels, columns = labels)


# **CONCLUSIONS ** the confusion matrix using entropy, and gini are the same. This means that their classification metrics will be equal. 
# For the iris dataset we are working with there does not seem to be a significant difference between the two measures of impurity

# Look at the classification report.
t = pd.DataFrame(classification_report(y_train, y_pred, output_dict = True))
t.loc['accuracy', :]

# 3. Print and clearly label the following: Accuracy, true positive rate, false positive rate, true negative rate, false negative rate, precision, recall, f1-score, support.
# From the confusion matrix (cmat) calculate:
# True Positive:
TP = np.diag(cmat).sum()

# False Positive:
FP = (cmat.sum(axis = 0) - np.diag(cmat)).sum()

# False Negative:
FN = (cmat.sum(axis = 1) - np.diag(cmat)).sum()

# True Negative:
TN = 2* cmat.values.sum() - FP

# True positive rate/ Recall
TPR = TP/(TP+FN)

# True negative rate
TNR = TN/(TN+FP)

# Precision
PPV = TP/(TP+FP)

# Negative predictive value
NPV = TN/(TN+FN)

# False positive rate
FPR = FP/(FP+TN)

# False negative rate
FNR = FN/(TP+FN)

# Overall accuracy
ACC = (TP + TN)/(TP+FP+FN+TN)

# F-1 Score
f1_score = 2 * ((TPR * PPV)/(TPR + PPV))

# support
support = len(y_train)

# Create a dataframe of classification metrics:
cmet = pd.DataFrame(({'classification_metrics': 
                     {'accuracy': ACC,
                      'true_positive_rate': TPR,
                      'false_positive_rate': FPR,
                      'true_negative_rate': TNR,
                      'false_negative_rate': FNR,
                      'precision': PPV,
                      'recall': TPR,
                      'f1_score': f1_score,
                      'support' : support}}))

dot_data = export_graphviz(clf, out_file = None)
graph = graphviz.Source(dot_data)

graph.render('iris_decsion_tree', view = True)


# RANDOM FOREST

# Continue working in your model file.

from sklearn.ensemble import RandomForestClassifier

# 1. Fit the Random Forest classifer to your training sample and transform (i.e. make predictions on the training sample) setting the random_state accordingly and setting min_samples_leaf = 1 and max_depth = 20.
# 4. Run through steps increasing your min_samples_leaf to 5 and decreasing your max_depth to 3.

# Create the RandomForestClassifier object
rf = RandomForestClassifier(bootstrap = True, 
                            class_weight = None,
                            min_samples_leaf = 1,
                            n_estimators = 100,
                            max_depth = 20,
                            random_state = 123)

rf2 = RandomForestClassifier(bootstrap = True,
                             class_weight = None,
                             min_samples_leaf = 5,
                             n_estimators = 100,
                             max_depth = 3,
                             random_state = 123)

# Fit the object to the training data

rf.fit(X_train,y_train)
rf2.fit(X_train,y_train)

# Print Feature Importances:
rf.feature_importances_
rf2.feature_importances_

# Estimate the types of species using the training data

y_pred = rf.predict(X_train)
y_pred_2 = rf2.predict(X_train)

# Estimate the propability of each type of speicies prediction, using the training data.

y_pred_proba = rf.predict_proba(X_train)
y_pred_proba_2 = rf2.predict_proba(X_train)

# 2. Evaluate your results using the model score, confusion matriz, and classification report.

model_score = rf.score(X_train,y_train)
cr = pd.DataFrame(classification_report(y_train, y_pred, output_dict = True))
cmat = pd.DataFrame(confusion_matrix(y_train,y_pred),index = labels, columns = labels)

model_score_2 = rf2.score(X_train,y_train)
cr_2 = pd.DataFrame(classification_report(y_train, y_pred_2, output_dict = True))
cmat_2 = pd.DataFrame(confusion_matrix(y_train, y_pred_2), index = labels, columns = labels)
# 3. Print and clearly label the following: Accuracy, true positive rate, false positive rate, true negative rate, false negative rate, precision, recall, f1-score, support.
# From the confusion matrix (cmat) calculate:
# True Positive:
TP = np.diag(cmat).sum()

# False Positive:
FP = (cmat.sum(axis = 0) - np.diag(cmat)).sum()

# False Negative:
FN = (cmat.sum(axis = 1) - np.diag(cmat)).sum()

# True Negative:
TN = 2* cmat.values.sum() - FP

# True positive rate/ Recall
TPR = TP/(TP+FN)

# True negative rate
TNR = TN/(TN+FP)

# Precision
PPV = TP/(TP+FP)

# Negative predictive value
NPV = TN/(TN+FN)

# False positive rate
FPR = FP/(FP+TN)

# False negative rate
FNR = FN/(TP+FN)

# Overall accuracy
ACC = (TP + TN)/(TP+FP+FN+TN)

# F-1 Score
f1_score = 2 * ((TPR * PPV)/(TPR + PPV))

# support
support = len(y_train)

# Create a dataframe of classification metrics:
cmet = pd.DataFrame(({'classification_metrics': 
                     {'accuracy': ACC,
                      'true_positive_rate': TPR,
                      'false_positive_rate': FPR,
                      'true_negative_rate': TNR,
                      'false_negative_rate': FNR,
                      'precision': PPV,
                      'recall': TPR,
                      'f1_score': f1_score,
                      'support' : support}}))


# KNN 

# Continue working in your model notebook or python script.

# 1. Fit a K-Nearest Neighbors classifier to your training sample and transform (i.e. make predictions on the training sample)
# 4. Run through setting k to 10
# 5. Run through setting k to 20

from sklearn.neighbors import KNeighborsClassifier

# k = 5
knn = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform')

# k = 10
knn_10 = KNeighborsClassifier(n_neighbors=10, weights='uniform')

# k = 20
knn_20 = KNeighborsClassifier(n_neighbors=20, weights='uniform')

# Fit for k = 5
knn.fit(X_train, y_train)

# Fit for k = 10
knn_10.fit(X_train, y_train)

# Fit for k = 20
knn_20.fit(X_train, y_train)

# Predict k = 5
y_pred = knn.predict(X_train)

# Predict k = 10
y_pred_10 = knn_10.predict(X_train)

# Predict k = 20
y_pred_20  = knn_20.predict(X_train)

# Predict probability k = 5
y_pred_proba = knn.predict_proba(X_train)

# Predict probability k = 10
y_pred_proba_10 = knn_10.predict_proba(X_train)

# Predict probability k = 20
y_pred_proba_20 = knn_20.predict_proba(X_train)


# Look at the score of the model, k = 5
knn.score(X_train, y_train)

# Look at the score of the model, k = 10
knn_10.score(X_train, y_train)

#Look at the score of the model, k = 20
knn_20.score(X_train, y_train)

# Create a classification report k = 5
model_score = knn.score(X_train,y_train)
cr = pd.DataFrame(classification_report(y_train, y_pred, output_dict = True))

# Create a classification report k = 10
cr_10 = pd.DataFrame(classification_report(y_train, y_pred_10, output_dict = True))

# Create a classification report k = 20
cr_20 = pd.DataFrame(classification_report(y_train, y_pred_20, output_dict = True))

# Create a confusion matrix k = 5
cmat = pd.DataFrame(confusion_matrix(y_train,y_pred),index = labels, columns = labels)

# Create a confusion matrix k = 10
cmat_10 = pd.DataFrame(confusion_matrix(y_train, y_pred_10), index = labels, columns = labels)

# Create a confusion matrix k = 20
cmat_20 = pd.DataFrame(confusion_matrix(y_train, y_pred_20), index = labels, columns = labels)

# 3. Print and clearly label the following: Accuracy, true positive rate, false positive rate, true negative rate, false negative rate, precision, recall, f1-score, support.
# From the confusion matrix (cmat) calculate:

# True Positive (k = 5):
def TP(cmat):
    return np.diag(cmat).sum()

TP_5 = TP(cmat)

# True Positive (k = 10):
TP_10 = TP(cmat_10)

# True Positive (k = 20):
TP_20 = TP(cmat_20)

# False Positive (k = 5):
FP = (cmat.sum(axis = 0) - np.diag(cmat)).sum()

# False Positive (k = 10):
FP_10 = (cmat_10.sum(axis = 0) - np.diag(cmat_10)).sum()

# False Positive (k = 20):
FP_20 = (cmat_20.sum(axis = 0) - np.diag(cmat_10)).sum()

# False Negative (k = 5):
FN = (cmat.sum(axis = 1) - np.diag(cmat)).sum()

# False Negative (k =10):
FN_10 = (cmat_10.sum(axis = 1) - np.diag(cmat_10)).sum()

# False Negative (k = 20):
FN_20 = (cmat_20.sum(axis = 1) - np.diag(cmat_20)).sum()

# True Negative (k = 5):
TN = 2* cmat.values.sum() - FP

# True Negative (k = 10):
TN_10 = 2 * cmat_10.values.sum() - FP_10

# True Negative (k = 20):
TN_20 = 2 * cmat_20.values.sum() - FP_20

# True positive rate/ Recall (k = 5):
TPR = TP/(TP+FN)

# True positive rate/ Recall (k =10):
TPR_10 = TP_10/(TP_10 + FN_10)

# True positive rate/ Recall (k = 20):
TPR_20 = TP_20/(TP_20 + FN_20)

# True negative rate (k = 5):
TNR = TN/(TN+FP)

# True negative rate (k = 10):
TNR_10 = TN_10/(TN_10 + FP_10)

# True negative rate (k = 20):
TNR_20 = TN_20/(TN_20 + FP_20)

# Precision (k = 5):
PPV = TP/(TP+FP)

# Precision (k = 10):
PPV_10 = TP_10/(TP_10 + FP_10)

# Precision (k = 20):
PPV_20 = TP_20/(TP_20 + FP_20)

# Negative predictive value (k = 5):
NPV = TN/(TN+FN)

# Negative predictive value (k = 10):
NPV_10 = TN_10/(TN_10+FN_10)

# Negative predictive value (k = 20):
NPV_20 = TN_20/(TN_20+FN_20)

# False positive rate (k = 5):
FPR = FP/(FP+TN)

# False positive rate (k = 10):
FPR_10 = FP_10/(FP_10 + TN_10)

# False positive rate (k = 20):
FPR_20 = FP_20/(FP_20 + TN_20)

# False negative rate (k = 5):
FNR = FN/(TP+FN)

# False negative rate (k = 10):
FNR_10 = FN_10/(TN_10 + FN_10)

# False negative rate (k = 20):
FNR_20 = FN_20/(TN_20 + FN_20)

# Overall accuracy (k = 5):
ACC = (TP + TN)/(TP+FP+FN+TN)

# Overall accuracy (k = 10):
ACC_10 = (TP_10 + TN_10)/ (TP_10 + FP_10 + FN_10 + TN_10)

# Overall accuracy (k = 20):
ACC_20 = (TP_20 + TN_20)/(TP_20 + FP_20 + FN_20 + TN_20)

# F-1 Score (k = 5):
f1_score = 2 * ((TPR * PPV)/(TPR + PPV))

# F-1 Score (k = 10):
f1_score_10 = 2 * ((TPR_10 * PPV_10)/(TPR_10 + PPV_10))

# F-1 Score (k = 20):
f1_score_20 = 2 * ((TPR_20 * PPV_20)/(TPR_20 + PPV_20))

# support
support = len(y_train)

# Create a dataframe of classification metrics:
cmet = pd.DataFrame(({'classification_metrics': 
                     {'accuracy': ACC,
                      'true_positive_rate': TPR,
                      'false_positive_rate': FPR,
                      'true_negative_rate': TNR,
                      'false_negative_rate': FNR,
                      'precision': PPV,
                      'recall': TPR,
                      'f1_score': f1_score,
                      'support' : support}}))

cmet_10 = pd.DataFrame(({'classification_metrics_10': 
                     {'accuracy': ACC_10,
                      'true_positive_rate': TPR_10,
                      'false_positive_rate': FPR_10,
                      'true_negative_rate': TNR_10,
                      'false_negative_rate': FNR_10,
                      'precision': PPV_10,
                      'recall': TPR_10,
                      'f1_score': f1_score_10,
                      'support' : support}}))

cmet_20 = pd.DataFrame(({'classification_metrics_20': 
                     {'accuracy': ACC_20,
                      'true_positive_rate': TPR_20,
                      'false_positive_rate': FPR_20,
                      'true_negative_rate': TNR_20,
                      'false_negative_rate': FNR_20,
                      'precision': PPV_20,
                      'recall': TPR_20,
                      'f1_score': f1_score_10,
                      'support' : support}}))

cmet.join([cmet_10, cmet_20])

# ** TEST ***

# For both the iris and titantic data,

# 1. Determine which model(with hyperparameters) performs the best (try reducing the number of features to the top 4 features in terms of information gained for each feature individually.)

train, test, int_encoder = prep_titanic()

train.head()
train.alone.value_counts()
train.sibsp.value_counts().sort_index()
train.parch.value_counts()

df = train[pd.notna(train.age)]
df.head()

df.survived.value_counts().sort_index().plot(kind = "bar", alpha = .5)
plt.title("Distribution of 'passengers survived")
plt.grid(b = True, which = "major")

features = ['sex', 'class', 'embarked_encoded', 'alone']

_, ax = plt.subplots(nrows = 1, ncols = 4, figsize = (18,5))

survival_rate = df.survived.mean()

for i, feature in enumerate(features):
    sns.barplot(feature, 'survived', data = df, ax = ax[i], alpha = .5)
    ax[i].set_ylabel('Survival Rate')
    ax[i].axhline(survival_rate, ls = '--', color = 'grey')

# RFE selection:
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder


df.sex.value_counts(dropna = False)
encoder = LabelEncoder()
df.sex = encoder.fit_transform(df.sex)
df.sex.head()
df.head()

cols = list(X_train.columns)
model = LogisticRegression()

X_train = df.drop(columns = ['survived', 'embarked', 'class', 'embark_town'])
y_train = df[['survived']]

rfe = RFE(model, 4)

# transforming data using RFE
X_rfe = rfe.fit_transform(X_train, y_train)

#Fitting the data to model
model.fit(X_rfe, y_train)
temp = pd.Series(rfe.support_, index = cols)


X_train = X_train[['pclass', 'sex', 'age', 'fare']]

model.fit(X_train, y_train)
model.score(X_train, y_train)

y_pred = model.predict(X_train)
confusion_matrix(y_train, y_pred)

