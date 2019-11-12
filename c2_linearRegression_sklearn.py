import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
boston=datasets.load_boston()

X=boston.data
y=boston.target

X=X[y<50.0]
y=y[y<50.0]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=666)
li=LinearRegression()
li.fit(X_train,y_train)
print(li.coef_)
print(li.intercept_)
print(li.score(X_test,y_test))

#knn 回归,网格搜索
from sklearn.neighbors import KNeighborsRegressor

knn=KNeighborsRegressor()
knn.fit(X_train,y_train)