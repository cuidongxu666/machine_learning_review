import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

#线性SVM,决策边界是直线（数据点线性可分）
from sklearn.svm import LinearSVC
iris=datasets.load_iris()
X=iris.data
y=iris.target
X=X[y<2,:2]
y=y[y<2]


#非线性SVM，决策边界曲线（数据点线性不可分）
from sklearn.svm import SVC