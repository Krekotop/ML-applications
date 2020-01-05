# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 22:55:50 2019

@author: User
"""

import mglearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
X,y = mglearn.datasets.load_extended_boston()
ridge = Ridge().fit(X_train,y_train)
print("Training set score:{:.2f}".format(ridge.score(X_train,y_train)))
print("Test set score:{:.2f}".format(ridge.score(X_test,y_test)))