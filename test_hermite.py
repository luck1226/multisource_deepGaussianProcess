
import numpy as np
from matplotlib import pyplot as plt

def f(x):
    return 0.5**(0.5) + (1/np.pi)**(0.5)*x - (1/6/np.pi)**(0.5)*(x**3-3*x) + (3/40/np.pi)**(0.5)*(x**5-10*x**3+15*x)

x = np.linspace(-1,1,500)[:,None]
y = f(x)

plt.plot(x,y,'k-')
plt.show()
