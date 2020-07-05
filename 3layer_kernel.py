
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

#def f(x,c):
#    return (np.sqrt(1+2.*c**2.*(1.-x)))**(-1)


def k_rbf(x,c):
    return np.exp((-1)*x**2/2/c**2)

def k1(x,c):
    return (1-np.exp((-1.)*x**2/2/c**2))

def k2(x,c1,c2):
    tmp1 = np.sqrt(k1(x,c1)) * np.sqrt(2) * c2
    tmp2 = tmp1**(-1)
    return erf(tmp2)

def k3(x,c1,c2,c3):
    tmp1 = 1/np.sqrt(1+2*c3**2)
    #return tmp1*k2((-1.)*x,c1,c2)+k2(x,c1,c2)
    return 1/np.sqrt(1+2*c3**2)+(1-1/np.sqrt(1+2*c3**2))*k2(x,c1,c2)

def k_eff1(x,c1,c2):
    return (np.sqrt(1+2*c2**2*k1(x,c1)))**(-1)

def k_eff2(x,c1,c2,c3):
    return (np.sqrt(1+2*c3**2*(1-k_eff1(x,c1,c2))))**(-1)


SIZE = 80

x_train = np.linspace(0, 8, SIZE)
x_train_b = np.linspace(0, 8, SIZE)[:, np.newaxis]

kk = x_train-x_train_b

#K_rbf = k_rbf(kk, 1)
#K_2 = k_eff1(kk, 1, 1)
#K_3 = k_eff2(kk, 1, 1, 1)
#K_33 = k3(kk, 1, 1, 1)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#fig.suptitle('Sharing x per column, y per row')
#ax1.plot(x, y)
#ax2.plot(x, y**2, 'tab:orange')
#ax3.plot(x, -y, 'tab:green')
#ax4.plot(x, -y**2, 'tab:red')


ax1.plot(x_train, k3(x_train,1,1,1),'k-')
ax1.plot(x_train, k3(x_train,1,1.2,1.4), 'r-')
ax1.plot(x_train, k3(x_train,1,0.8,0.8), 'g-')
ax1.text(7, 0.95, '(a)', style='italic')
ax1.set_ylim([0.55,1.05])

ax2.plot(x_train, k_eff2(x_train,1,1,1),'k--')
ax2.plot(x_train, k_eff2(x_train,1,1.2,1.4), 'r--')
ax2.plot(x_train, k_eff2(x_train,1,0.8,0.8), 'g--')
ax2.text(7, 0.95, '(b)', style='italic')
ax2.set_ylim([0.55,1.05])
#plt.plot(x_train, k_rbf(x_train, 1), 'k-')
#plt.plot(x_train, k_eff1(x_train, 1, 1), 'r-')
#plt.plot(x_train, k_eff2(x_train, 1, 1, 1), 'b-')
#plt.plot(x_train, k3(x_train, 1, 1, 1), 'b--')
#plt.xlabel(r'$|x_i-x_j|$')
#plt.ylabel(r'$k(x_i,x_j)$')

K_3a = k_eff2(kk, 1, 1, 1)
K_3b = k_eff2(kk, 1, 1.2, 1.4)
K_3c = k_eff2(kk, 1, 0.8, 0.8)

K_33a = k3(kk, 1, 1, 1)
K_33b = k3(kk, 1, 1.2, 1.4)
K_33c = k3(kk, 1, 0.8, 0.8)
#plt.plot(x_train, np.random.multivariate_normal(np.zeros(SIZE), K_rbf, 1).T, 'y-')
#plt.plot(x_train, np.random.multivariate_normal(np.zeros(SIZE), K_2, 2).T, 'g-')

#np.random.seed(31)

ax4.plot(x_train, np.random.multivariate_normal(np.zeros(SIZE), K_3a, 1).T, 'k--')
ax4.plot(x_train, np.random.multivariate_normal(np.zeros(SIZE), K_3b, 1).T, 'r--')
ax4.plot(x_train, np.random.multivariate_normal(np.zeros(SIZE), K_3c, 1).T, 'g--')
ax4.set_ylim([-2.5, 2.5])
ax4.text(7, 1.5, '(d)', style='italic')

ax3.plot(x_train, np.random.multivariate_normal(np.zeros(SIZE), K_33a, 1).T, 'k-')
ax3.plot(x_train, np.random.multivariate_normal(np.zeros(SIZE), K_33b, 1).T, 'r-')
ax3.plot(x_train, np.random.multivariate_normal(np.zeros(SIZE), K_33c, 1).T, 'g-')
ax3.set_ylim([-2.5, 2.5])
ax3.text(7, 1.5, '(c)', style='italic')
plt.show()






