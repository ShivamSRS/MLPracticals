# Feature Extraction with PCA
import numpy
from pandas import read_csv
from sklearn.decomposition import PCA
# load data
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] #feature names
dataframe = read_csv('pima.csv', names=names)#diabetes dataset
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
pca = PCA(n_components=6)
pca.fit(X)
# summarize components
print("Explained Variance: %s" % pca.explained_variance_ratio_)
print(pca.components_)
"""You can see that the transformed dataset (6 principal components) bare little resemblance to the source data.
total 6 features now instead of 8"""

from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
# feature extraction
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X[:(3*len(X)//4)],Y[:(3*len(X)//4)])
y_pred = model.predict(X[(3*len(X)//4):])
print("Using 8 features")
print("Accuracy:",metrics.accuracy_score(Y[(3*len(X)//4):], y_pred))

X2 = pca.transform(X)

model2 = ExtraTreesClassifier(n_estimators=10)
model2.fit(X2[:(3*len(X2)//4)],Y[:(3*len(X2)//4)])
y_pred2 = model2.predict(X2[(3*len(X2)//4):])
print("Using PCA features")
print("Accuracy:",metrics.accuracy_score(Y[(3*len(X2)//4):], y_pred2))