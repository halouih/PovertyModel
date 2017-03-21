# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 18:20:26 2017

@author: Hamza
"""

from sklearn import tree
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

