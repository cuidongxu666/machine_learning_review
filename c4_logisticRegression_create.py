import numpy as np
from c1_metrics import accuracy_score
#本小节不实现随机梯度下降法，随机梯度也可实现
class LogisticRegression:
    def __init__(self):
        self.coef_=None
        self.interception_=None
        self._theta=None

    def _sigmoid(self,t):
        return 1./(1.+np.exp(-t))

    #批量梯度下降求解，梯度的计算式一项一项的
    def fit_gd(self,X_train,y_train,eta=0.01,n_iter=1e4):
        assert X_train.shape[0]==y_train.shape[0],''

        def J(theta,X_b,y):
            y_hat=self._sigmoid(X_b.dot(theta))
            try:
                return -np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))/len(y)
            except:
                return float('inf')
        #向量化：每项点乘与综合考虑
        def dJ(theta,X_b,y):
            # res=np.empty((len(theta),))
            # res[0]=np.sum(X_b.dot(theta)-y)
            # for i in range(1,len(theta)):
            #     res[i]=(X_b.dot(theta)-y).dot(X_b[:,i])
            # return res*2/len(X_b)
            return X_b.T.dot(self._sigmoid(X_b.dot(theta))-y)/len(y)

        def gradient_descent(X_b,y,initial_theta,eta,n_inter=1e4,epsilon=1e-8):
            theta=initial_theta
            cur_iter=0
            while cur_iter<n_inter:
                gradient=dJ(theta,X_b,y)
                last_theta=theta
                theta=theta-eta*gradient
                if (abs(J(theta,X_b,y)-J(last_theta,X_b,y))<epsilon):
                    break
                cur_iter+=1
            return theta

        X_b=np.hstack([np.ones((len(X_train),1)),X_train])
        initial_theta=np.zeros(X_b.shape[1])
        self._theta=gradient_descent(X_b,y_train,initial_theta,eta,n_iter)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self
    #概率 向量
    def predict_proba(self,X_predict):
        assert self.interception_ is not None and self.coef_ is not None,''
        assert X_predict.shape[1] ==len(self.coef_),''

        X_b=np.hstack([np.ones((len(X_predict),1)),X_predict])
        return self._sigmoid(X_b.dot(self._theta))
    #类别 向量
    def predict(self,X_predict):
        assert self.interception_ is not None and self.coef_ is not None,''
        assert X_predict.shape[1] ==len(self.coef_),''

        proba=self.predict(X_predict)
        #bull强制转换成int
        return np.array(proba >=0.5,dtype='int')
    #准确率
    def score(self,X_test,y_test):
        y_predict=self.predict(X_test)
        return accuracy_score(y_test,y_predict)

