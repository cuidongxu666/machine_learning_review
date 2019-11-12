import numpy as np
from collections import Counter
from metrics import accuracy_score
class KNNClassifer:
    def __init__(self,k):
        assert k>=1,'k must be valid'
        self.k=k
        self._X_train=None
        self._y_train=None

    def fit(self,X_train,y_train):
        assert X_train.shape[0]==y_train.shape[0],'训练集样本数必须等于标记个数'
        assert  self.k <= X_train.shape[0],''
        self._X_train=X_train
        self._y_train=y_train
        return self#链式调用

    def predict(self,X_predict):#传入array
        assert self._X_train is not None and self._y_train is not None,''
        assert X_predict.shape[1]==self._X_train.shape[1],''

        y_predict=[self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self,x):
        assert x.shape[0] == self._X_train.shape[1],''

        distances=[np.sqrt(np.sum((x_train-x)**2)) for x_train in self._X_train]
        nearest=np.argsort(distances)
        top_y=[self._y_train[i] for i in nearest[:self.k]]
        votes=Counter(top_y)
        return votes.most_common(1)[0][0]

    def score(self,X_test,y_test):
        y_predict=self.predict(X_test)
        return accuracy_score(y_test,y_predict )


