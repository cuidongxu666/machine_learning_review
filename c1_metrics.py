import numpy as np
def accuracy_score(y_true,y_predict):
    assert y_true[0]==y_predict.shape[0],''
    return sum(y_predict==y_true)/len(y_true)

def mean_squard_error(y_true,y_predict):
    assert len(y_true)==len(y_predict),''
    return np.sum((y_true-y_predict)**2)/len(y_true)

def root_mean_squared_error(y_true,y_predict):
    return np.sqrt(mean_squard_error(y_true,y_predict))

def mean_absolute_error(y_true,y_predict):
    assert len(y_true)==len(y_predict),''
    return np.sum(np.absolute(y_predict-y_true))/len(y_true)

def r2_score(y_true,y_predict):
    return 1-mean_squard_error(y_true,y_predict)/np.var(y_true)