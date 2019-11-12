from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
# iris=datasets.load_iris()
# X=iris.data
# y=iris.target
#
# X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=666)
# knn=KNeighborsClassifier(n_neighbors=6)
# knn.fit(X_train,y_train)
# y_predict=knn.predict(X_test)
# print(y_predict)
# -----------------------------------------------
#网格搜索+交叉验证(n_jobs,verbose=2(输出状态，值越大，越详细 ))
digits=datasets.load_digits()
X=digits.data
y=digits.target
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=666)
#数据均值方差归一化
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(X_train)
X_train=ss.transform(X_train)
X_test_predict=ss.transform(X_test)
#网格搜索
param_grid=[
    {
        'weights':['uniform'],
        'n_neighbors':[i for i in range(1,5)]
    },
    {
        'weights':['distance'],
        'n_neighbors':[i for i in range(1,5)],
        'p':[i for i in range(1,6)]
    }
]

knn=KNeighborsClassifier()
grid_search=GridSearchCV(knn,param_grid)
grid_search.fit(X_train,y_train)
knn_best=grid_search.best_estimator_#最好分类器，直接拿来用
y_predict=knn_best.predict(X_test_predict)
