# -*- coding: utf-8 -*-
"""
SVM on forge dataset(nonlinear boundary)
"""
import mglearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.svm import SVC
X , y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel = 'rbf',C=10,gamma = 0.1).fit(X,y)
mglearn.plots.plot_2d_separator(svm,X,eps=.5)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
sv = svm.support_vectors_
sv_labels = svm.dual_coef_.ravel() >0
mglearn.discrete_scatter(sv[:,0],sv[:,1],sv_labels,s=15,markeredgewidth = 3)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
