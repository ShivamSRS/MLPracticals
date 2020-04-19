# Bagged Decision Trees for Classification
import pandas
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
dataframe = pandas.read_csv("pima.csv")
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 69
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X[:(len(X)//2)],Y[:(len(X)//2)])

#Predict the response for test dataset
y_pred = clf.predict(X[(len(X)//2):])
print("Using simple Decision Tree")
print("Accuracy:",metrics.accuracy_score(Y[(len(X)//2):], y_pred))
print("-------------------")

##voting ensemble with decision tree n logistic regression
from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X,Y, cv=kfold)
print("voting Accuracy is",results.mean())
print("-------------------")

##
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("BaggingClassifier Accuracy is ",results.mean())
print("-------------------")

#random forest
from sklearn.ensemble import RandomForestClassifier
num_trees = 100
max_features = 3
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("random forest Accuracy is",results.mean())
print("-------------------")

#Adaboost
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("adaboost Accuracy is",results.mean())
print("-------------------")
#Gradient boosting
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Gradient boosting",results.mean())
print("-------------------")