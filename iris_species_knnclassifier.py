# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 23:02:07 2019ica
There are three types of species of iris flowers to specify, namely setosa(coded as 0),versocolor(1),virginica(2). 
Each of them has its own featuress specified, namely sepal length (cm), sepal width (cm), petal length (cm), petal width (cm). Based on given samples,
for which names of certain combinations of these features have their names already classified, we are going to write a simple program to classify any given new species
to the one of three classes(using k-Nearest Neighbors)

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
 
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
iris_dataframe = pd.DataFrame(X_train, columns = iris_dataset.feature_names)
#grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15),marker = 'o',hist_kwds = {'bins': 20}, s = 60,alpha = .8,cmap=mglearn.cm3)
""" we see how well separated species through these measurements.It gives us some confidence in positive result"""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
print("Test set score:{:.2f}".format(knn.score(X_test,y_test)))
X_new = []
X_new.append(input("What is the sepal length?\n"))
X_new.append(input("What is the sepal width?\n"))
X_new.append(input("What is the petal length?\n"))
X_new.append(input("What is the petal width?\n"))
X_newt = np.array([X_new])
prediction = knn.predict(X_newt)
print("Prediction:{}".format(prediction))
print("Predicted target name:{}".format(iris_dataset['target_names'][prediction]))