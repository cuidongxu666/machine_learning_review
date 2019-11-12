#模拟信息熵进行一次划分
from collections import Counter
import numpy as np
from math import log
#根据特征d，阈值value进行划分
def split(X,y,d,value):
    index_a=(X[:d]<=value)
    index_b=(X[:d]>value)
    return X[index_a],X[index_b],y[index_a],y[index_b]

#计算信息熵  基尼系数
def entropy(y):
    counter=Counter(y)
    res=0.0
    #res=1
    for num in counter.values():
        p=num/len(y)
        res+=-p*log(p)
        #res-=p**2

def try_split(X,y):
    best_entropy=float('inf')
    best_d,best_v=-1,-1
    for d in range(X.shape[1]):
        sorted_index=np.argsort(X[:d])
        for i in range(1,len(X)):
            if X[sorted_index[i-1],d]!=X[sorted_index[i],d]:
                v=(X[sorted_index[i-1],d]+X[sorted_index[i],d])/2
                X_l,X_r,y_l,y_r=split(X,y,d,v)
#重点：划分后，对划分后的两组分别求信息熵，相加（划分后的整体信息熵）
                e=entropy(y_l)+entropy(y_r)
                if e<best_entropy:
                    best_entropy,best_d,best_v=e,d,v
    return best_entropy,best_d,best_v

