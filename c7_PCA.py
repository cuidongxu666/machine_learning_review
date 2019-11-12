import numpy as np
def demean(X):
    return X-np.mean(X,axis=0)

#梯度上升
def f(w,X):
    return np.sum((X.dot(w)**2))/len(X)

def df_math(w,X):
    return X.T.dot(X.dot(w))*2./len(X)

#把向量搞成单位向量
def direction(w):
    return w/np.linalg.norm(w)