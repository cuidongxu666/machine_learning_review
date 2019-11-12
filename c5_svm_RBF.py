#直观理解高斯核函数,映射到二维
import numpy as np
import matplotlib.pyplot as plt
x=np.arange(-4,5,1)
y=np.array((x>=-2)&(x<=2),dtype='int')

# plt.scatter(x[y==0],[0]*len(x[y==0]))
# plt.scatter(x[y==1],[0]*len(x[y==1]))
# plt.show()

def gs(x,l):
    gamma=1.0
    return np.exp(-gamma*(x-l)**2)

l1,l2=-1,1

X_new=np.empty((len(x),2))
for i,data in enumerate(x):
    X_new[i,0]=gs(data,l1)
    X_new[i,1]=gs(data,l2)

plt.scatter(X_new[y==0,0],X_new[y==0,1])
plt.scatter(X_new[y==1,0],X_new[y==1,1])
plt.show()


