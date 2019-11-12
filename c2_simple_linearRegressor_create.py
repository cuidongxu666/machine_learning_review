import numpy as np
from c1_metrics import r2_score
class SimpleLinearRegression1:
    def __init__(self):
        self.a_=None
        self.b_=None
    #公式简单运算
    def fit1(self,x_train,y_train):
        assert x_train.ndim==1,''
        assert len(x_train)==len(y_train),''

        x_mean=np.mean(x_train)
        y_mean=np.mean(y_train)

        num=0.0
        d=0.0
        for x,y in zip(x_train,y_train):
            num+=(x-x_mean)*(y-y_mean)
            d+=(x-x_mean)**2
        self.a_=num/d
        self.b_=y_mean-self.a_*x_mean
        return self
    #公式向量化运算(性能大幅 提升)
    def fit2(self,x_train,y_train):
        assert x_train.ndim==1,''
        assert len(x_train)==len(y_train),''

        x_mean=np.mean(x_train)
        y_mean=np.mean(y_train)

        num=(x_train-x_mean).dot(y_train-y_mean)
        d=(x_train-x_mean).dot(x_train-x_mean)

        self.a_=num/d
        self.b_=y_mean-self.a_*x_mean
        return self

    def predict(self,x_predict):#传进数组
        assert x_predict.ndim==1,''
        assert self.a_ is not None and self.b_ is not None,''
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self,x):
        return self.a_*x +self.b_

    def score(self,x_test,y_test):
        y_predict=self.predict(x_test)
        return r2_score(y_test,y_predict)