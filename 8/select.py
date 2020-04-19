"""
Filter based: We specify some metric and based on that filter features.
 An example of such a metric could be correlation/chi-square.

Wrapper-based: Wrapper methods consider the selection of a set of features as a search problem. 
Example: Recursive Feature Elimination

Embedded: Embedded methods use algorithms that have built-in feature selection methods. 
For instance, Lasso and RF have their own feature selection methods.
"""

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
import numpy as np 
dataframe = read_csv('pima.csv')
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature selection
print("Filter method")
#filter method
test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, Y)
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
print("feature names are",names)
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])

print("one can see the scores for each attribute and the 4 attributes chosen (those with the highest scores)")
print("Specifically features with indexes 0 (preq), 1 (plas), 5 (mass), and 7 (age).")
print("##########","wrapper method",sep="\n")
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

print("You can see that RFE chose the the top 3 features as preg, mass and pedi.")
print("These are marked True in the support_ array and marked with a choice “1” in the ranking_ array.")

print("################","Embedded method",sep ='\n')


# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, Y)
print("feature_importances_")
print(names,model.feature_importances_)
