# importing dependencies
from pandas import read_csv
from pandas.plotting import scatter_matrix as sm
import scipy as sp
from matplotlib import pyplot as pl
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier

# loading data
url = 'D:/MyCaptain/A.I/Assignment_4/iris.csv'
names = ['sepal-length', 'sepal-width', 'pepal-length', 'petal-width', 'class']
data = read_csv(url, names=names)

# dimensions of data
print(data.shape)

# seeing the data set
print(data.head(20))

# statistical summary
print(data.describe())

# class distribution
print(data.groupby('class').size())

# univariate plots and whisker plot
data.plot(kind="box", subplots='True', layout=(2, 2), sharex=False, sharey=False)

# histogram of the variable
data.hist()
# pl.show()

# multivariate plots
sm(data)
# pl.show()

# creating a validation dataset
# splitting dataset
array = data.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)

# logistic regression
# linear discriminant analysis
# KNN
# classification and regression trees
# gaussian naive bayes
# support vector machine

# building models

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# compare our model
pl.boxplot(results, labels=names)
pl.title("Algorithm Comparision")
pl.show()

# make predictions on svm
model = SVC(gamma='auto')
model.fit(X_train, y_train)
prediction = model.predict(X_validation)

# evaluate our predictions
print(accuracy_score(y_validation,prediction))
print(confusion_matrix(y_validation,prediction))
print(classification_report(y_validation,prediction))
