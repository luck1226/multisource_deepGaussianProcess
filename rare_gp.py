
import numpy as np
import GPy
import seaborn as sns
from matplotlib import pyplot as plt


def f_low(x):
    #return np.sin(8*np.pi*x)
    #return x**2
    return np.cos(15.*x)

def f_high(x):
    #return (x-np.sqrt(2)) * (f_low(x))**2
    #return np.sin(2*np.pi*x**2)
    return x*np.exp(np.cos(30*x-3))-1


np.random.seed(59)

num_cheap = 30
num_expensive = 10

X1 = np.random.rand(num_cheap)[:,None]
X2 = np.random.rand(num_expensive)[:,None]
X_test = np.linspace(0,1,99)[:,None]

Ye = f_high(X2)
Y_true = f_high(X_test)

Kb = GPy.kern.RBF(input_dim = 1, variance = 0.5, lengthscale = 0.3)
mb = GPy.models.GPRegression(X2, Ye, Kb, noise_var = 0.0001)
mb.optimize(messages=True)
mb.optimize_restarts(num_restarts = 10)

mu, v = mb.predict(X_test)

plt.plot(X_test, mu, 'k-')
plt.plot(X_test, Y_true, 'r-')
plt.plot(X_test, mu+2*v, 'b--')
plt.plot(X_test, mu-2*v, 'b--')
plt.plot(X2, Ye,'ro')
plt.scatter(X2, Ye, color='r')

print(X2)

plt.show()

