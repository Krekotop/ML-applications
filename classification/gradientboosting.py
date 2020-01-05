# -*- coding: utf-8 -*-
"""
gradient boosting on cancer dataset
"""
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt 
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target, random_state=42)
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train,y_train)
print("Accuracy on trainin set:{:.3f}".format(gbrt.score(X_train,y_train)))
print("Accuracy on testi set:{:.3f}".format(gbrt.score(X_test,y_test)))
