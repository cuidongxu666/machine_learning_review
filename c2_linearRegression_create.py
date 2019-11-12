import numpy as np
from c1_metrics import r2_score
class LinearRegression:
    def __init__(self):
        self.coef_=None
        self.interception_=None
        self._theta=None
    #正规方程求解
    def fit_normal(self,X_train,y_train):

        assert X_train.shape[0]==y_train.shape[0],''
        X_b=np.hstack([np.ones((len(X_train),1)),X_train])
        self._theta=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_=self._theta[0]
        self.coef_=self._theta[1:]
        return self
    #批量梯度下降求解，梯度的计算式一项一项的
    def fit_gd(self,X_train,y_train,eta=0.01,n_iter=1e4):
        assert X_train.shape[0]==y_train.shape[0],''
        def J(theta,X_b,y):
            try:
                return np.sum((y-X_b.dot(theta))**2)/len(y)
            except:
                return float('inf')
        #向量化：每项点乘与综合考虑
        def dJ(theta,X_b,y):
            # res=np.empty((len(theta),))
            # res[0]=np.sum(X_b.dot(theta)-y)
            # for i in range(1,len(theta)):
            #     res[i]=(X_b.dot(theta)-y).dot(X_b[:,i])
            # return res*2/len(X_b)
            return X_b.T.dot(X_b.dot(theta)-y)*2./len(y)

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
    #随机梯度下降至少要把样本看一遍
    #随机梯度下降中的n_iters表示把所有样本看几圈
    def fit_sgd(self,X_train,y_train,n_iters=5,t0=5,t1=50):
        assert X_train.shape[0]==y_train.shape[0],''
        assert n_iters>=1,''
        def dJ_sgd(theta,X_b_i,y_i):
            return X_b_i*(X_b_i.dot(theta)-y_i)*2.

        def sgd(X_b,y,initial_theta,n_iters,t0=5,t1=50):

            def learning_rate(t):
                return t0/(t+t1)

            theta=initial_theta
            m=len(X_b)

            for cur_iter in range(n_iters):
                indexes=np.random.permutation(m)
                X_b_new=X_b[indexes]
                y_new=y[indexes]
                for i in range(m):
                    gradient=dJ_sgd(theta,X_b_new[i],y_new[i])
                    theta=theta-learning_rate(cur_iter*m+i)*gradient
            return theta

        X_b=np.hstack([np.ones((len(X_train),1),X_train)])
        initial_theta=np.random.randn(X_b.shape[1])
        self._theta=sgd(X_b,y_train,initial_theta,n_iters,t0,t1)
        self.interception_=self._theta[0]
        self.coef_=self._theta[1:]


    def predict(self,X_predict):
        assert self.interception_ is not None and self.coef_ is not None,''
        assert X_predict.shape[1] ==len(self.coef_),''

        X_b=np.hstack([np.ones((len(X_predict),1)),X_predict])
        return X_b.dot(self._theta)

    def score(self,X_test,y_test):
        y_predict=self.predict(X_test)
        return r2_score(y_test,y_predict)

