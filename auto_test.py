import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import GPy
from GPy.kern import Kern
from GPy import Param, Model

#from deepRBF import deepRBF
from deepRBF import deepCosine
from metric import *

auto_df = pd.read_csv('Auto.csv',na_values='?').dropna()

data = auto_df[['horsepower','displacement','weight','acceleration','mpg']].values

X = data[:,0:4]
y = data[:,-1].reshape(-1,1)

from sklearn.preprocessing import StandardScaler

X_stz = StandardScaler().fit_transform(X)
y_stz = StandardScaler().fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_stz, y_stz, test_size=0.2)

#d_rbf = deepRBF(input_dim=X_train.shape[1])
d_rbf = deepCosine(input_dim=X_train.shape[1])
m2 = GPy.models.GPRegression(X_train, y_train,d_rbf)
m2.optimize()

m2f_t, m2v_t = m2.predict(X_test)
m2f_tr, m2v_tr = m2.predict(X_train)

print("Test rmse: {}".format(rmse(m2f_t,y_test)))
print("Train rmse: {}".format(rmse(m2f_tr,y_train)))
print("Test NLL: {}".format(compute_nll(y_test,m2f_t,m2v_t)))
print("Train NLL: {}".format(compute_nll(y_train,m2f_tr,m2v_tr)))

#plt.scatter(m2f_t,y_test)
#plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
#plt.show()




