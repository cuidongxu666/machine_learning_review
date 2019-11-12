#多项式回归案例（自创特征）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x=np.random.uniform(-3,3,size=100)

X=x.reshape(-1,1)

y=0.5*x**2 + x+2 +np.random.normal(0,1,size=100)
# x_2=x**2
#
#
# x2=np.hstack([X,x_2.reshape(-1,1)])
#
#
# li=LinearRegression()
# li.fit(x2,y)
# y_predict=li.predict(x2)
#
# plt.scatter(x,y)
# plt.plot(np.sort(x),y_predict[np.argsort(x)],color='r')
# plt.show()




#sklearn 里多项式回归(升维，放在预处理中)
from sklearn.preprocessing import PolynomialFeatures
# poly=PolynomialFeatures(degree=2)
# poly.fit(X)
# #3列，第一列均为1，不需要去除，拟合时最后系数判为0
# X2=poly.transform(X)

from sklearn.linear_model import LinearRegression
# li2=LinearRegression()
# li2.fit(X2,y)
# theta=li2.coef_
# intercept=li2.intercept_



#管道
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
ploy_reg=Pipeline([
    ('ploy',PolynomialFeatures(degree=2)),
    ('std',StandardScaler()),
    ('linear',LinearRegression())
])
ploy_reg.fit(X,y)
y_predict2=ploy_reg.predict(X)
plt.scatter(x,y)
plt.plot(np.sort(x),y_predict2[np.argsort(x)],color='r')
plt.show()
