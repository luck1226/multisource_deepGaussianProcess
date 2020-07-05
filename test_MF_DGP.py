
import numpy as np
import GPy
from matplotlib import pyplot as plt


def f_low(x):
    return np.sin(8*np.pi*x)

def f_high(x):
    return (x-np.sqrt(2)) * (f_low(x))**2

def Cov_matrix_eff(mu, v, variance, lengthscale):
    m_eff = np.zeros((len(mu),len(mu)))
    for i in range(len(mu)):
        for j in range(len(mu)):
            if i == j:
                m_eff[i,j] = variance**2
            else:
                if m_eff[j,i] == 0:
                    d_mu = mu[i]-mu[j]
                    k_ii = v[i,i]
                    k_jj = v[j,j]
                    k_ij = v[i,j]
                    tmp_1 = 1 + (k_ii+k_jj-2*k_ij)/lengthscale**2
                    tmp_exp = np.exp((-1.)*d_mu**2/2/lengthscale**2/tmp_1)
                    #tmp_exp = 1.
                    tmp_sqrt = tmp_1**(-1/2)
                    m_eff[i,j] = variance**2 * tmp_exp * tmp_sqrt
                else:
                    m_eff[i,j] = m_eff[j,i]
    return m_eff

def predict_eff(CovMtx, Y):
    K_ee = CovMtx[1:,1:]
    K_e = CovMtx[0,1:]
    K_inv = np.linalg.inv(K_ee)
    pred_mu = np.matmul(np.matmul(K_e, K_inv), Y)
    pred_v = CovMtx[0,0] - np.matmul(np.matmul(K_e, K_inv), np.matrix.transpose(K_e))
    return pred_mu, np.sqrt(pred_v)


def SESE_fun(x, k00, var, ll):
    return var**2 * (1 + 2.*(k00 - x) / ll**2)**(-1/2)

def SESE_Cov_matrix(X, var, ll):
    k_test = GPy.kern.RBF(input_dim = 1, variance = var[0], lengthscale = ll[0])
    k_rbf = k_test.K(X)
    k00 = k_rbf[0][0]
    return SESE_fun(k_rbf, k00, var[1], ll[1])

def marginal_likelihood(Xc, Xe, Yc, Ye, var1, var2, len1, len2, var_noise, pathology):

    K1 = GPy.kern.RBF(input_dim = 1, variance = var1, lengthscale = len1)
    m1 = GPy.models.GPRegression(Xc, Yc, K1, noise_var = var_noise)
    mu1, v1 = m1.predict(Xe, full_cov = True)

    K2 = Cov_matrix_eff(mu1, v1, var2, len2)

    if pathology:
        K_aux = GPy.kern.RBF(1)
        Kse = K_aux.K(Xe)
        Keff = np.multiply(Kse, K2)
    else:
        Keff = K2

    K_inv = np.linalg.inv(Keff)

    tmp1 = 0.5 * np.matmul(np.matrix.transpose(Ye), np.matmul(K_inv, Ye))
    tmp2 = 0.5 * np.log(np.linalg.det(Keff))

    return tmp1+tmp2

def grid_optimize(Xc, Xe, Yc, Ye, var1, var2, var_noise, pathology):

    ax = np.linspace(0,1,50)
    bx = -2 + ax * 3
    cx1 = np.exp(bx)
    cx2 = np.exp(bx)

    marg = {}

    for ll1 in cx1:
        for ll2 in cx2:
            tmp = marginal_likelihood(Xc, Xe, Yc, Ye, var1, var2, ll1, ll2, var_noise, pathology)
            marg.update({(ll1,ll2):tmp})

    #print(marg)
    fav = min(marg, key=marg.get)
    return fav[0], fav[1]

def model_case_1(num_cheap, num_expensive, idx, pathology):

    var = 0.7

    np.random.seed(59)
    X1 = np.random.rand(num_cheap)[:,None]
    X2 = np.random.rand(num_expensive)[:,None]


    # idx = 1 for high-low fidelity
    # idx = 2 for high-low noise

    if idx is 1:
        Yc = f_low(X1)
    else:
        Yc = f_high(X1) + 0.1 * np.random.normal(0,1,num_cheap)[:,None]

    Ye = f_high(X2)

    K1 = GPy.kern.RBF(1)
    m1 = GPy.models.GPRegression(X1, Yc, K1, noise_var = 0.0001)
    m1.optimize(messages=True)
    m1.optimize_restarts(num_restarts = 10)

    mu, v = m1.predict(X2, full_cov = True)

    ax = np.linspace(0,1,50)
    bx = -2 + ax * 2.1
    cx = np.exp(bx)
    dx = -1 + ax * 1.5
    vx = np.exp(dx)

    mar_d = {}

    K_test = GPy.kern.RBF(1)
    kk_test = K_test.K(X2)

    for ll in cx:
        for vv in vx:


            MM = Cov_matrix_eff(mu, v, vv, ll)
            if pathology:
                GG = np.multiply(kk_test, MM)
            else:
                GG = MM

            GG = GG + 0.001 * np.identity(num_expensive)
            inv_GG = np.linalg.inv(GG)

            tmp1 = 0.5 * np.matmul(np.matrix.transpose(Ye), np.matmul(inv_GG, Ye))
            tmp2 = 0.5 * np.log(np.linalg.det(GG))

            mar_d.update({(vv,ll):(tmp1+tmp2)})


    fav = min(mar_d, key=mar_d.get)

    return fav


fav = model_case_1(40, 10, 2, 1)

#print(leng1)
print(fav)

#plt.figure(1)
#plt.plot(leng1, mar1, 'k-')

np.random.seed(59)

# index for applying pathology to kernel
pathh = 1

#vvaar1 = 1.0 is good for previous case
vvaar1 = 0.6
#vvaar2 = 0.5
vvaar2 = fav[0]

# 0.13 is good for the previous case
#lenl1 = 0.13

lenl1 = 0.1

#lenl2 = 0.6 is good for previous case
#lenl2 = 1.1
lenl2 = fav[1]

#X1 = np.random.rand(30)[:,None]
#X1 = np.linspace(0,1,30)[:,None]
X1 = np.random.rand(40)[:,None]
X2 = np.random.rand(10)[:,None]
#X2 = np.linspace(0,1,10)[:,None]

X_plot = np.linspace(0,1,200)[:,None]

X_test = np.linspace(0,1,99)[:,None]
f_c=f_low(X_plot)
f_e=f_high(X_plot)

#Y_c = f_low(X1)
Y_c = f_high(X1) + 0.1 * np.random.normal(0,1,len(X1))[:,None]
Y_e = f_high(X2)

#ss1, ss2 = grid_optimize(X1, X2, Y_c, Y_e, 1.0, 0.6, 0.0001, pathh)
#print(ss1)
#print(ss2)


K1 = GPy.kern.RBF(input_dim = 1, variance = vvaar1, lengthscale = lenl1)
# previous case with noise_var = 0.0001
m1 = GPy.models.GPRegression(X1, Y_c, K1, noise_var = 0.0001)
#m1.kern.variance.fix()
#m1.kern.noise_var.fix()
m1.optimize(messages=True)
m1.optimize_restarts(num_restarts = 10)
#m1.optimize()
print(m1)
fig = m1.plot()
GPy.plotting.show(fig)

Y_test = []
Y_test_up = []
Y_test_down = []

Y_sese = []

var_sese=[]
ll_sese=[]
var_sese.append(1.2)
var_sese.append(0.8)
ll_sese.append(0.15)
ll_sese.append(0.4)

K_test = GPy.kern.RBF(1)


for xx in X_test:
    Xnew = np.vstack((xx,X2))
    mu, v = m1.predict(Xnew, full_cov = True)
    print(mu)
    MM = Cov_matrix_eff(mu, v, vvaar2, lenl2)

    if pathh:
        kk_test = K_test.K(Xnew)
        GG = np.multiply(kk_test, MM)
    else:
        GG = MM

    #K_sese = SESE_Cov_matrix(Xnew, var_sese, ll_sese)


    #print(MM)
    pred_mu, pred_v = predict_eff(GG, Y_e)

    #mu_sese, v_sese = predict_eff(K_sese, Y_e)

    #print(pred_mu)
    Y_test.append(pred_mu)
    Y_test_up.append(pred_mu+2.*pred_v)
    Y_test_down.append(pred_mu-2.*pred_v)

    #Y_sese.append(mu_sese)


#print(Y_test)
#print(V_test)

#K2 = GPy.kern.RBF(input_dim=1,variance = 0.8, lengthscale = 0.2)
#m2 = GPy.models.GPRegression(X2, Y_e, K2, noise_var = 0.00001)
#m2.optimize(messages=True)
#m2.optimize_restarts(num_restarts = 10)
#print(m2)
#fig = m2.plot()
#GPy.plotting.show(fig)
#yy_test = []
#for xx in X_test:
#    mumu, vv = m2.predict(xx)
#    yy_test.append(mumu)

#plt.figure(2)

fig, ax = plt.subplots(1)

ax.plot(X1,Y_c,'k.')
ax.plot(X2,Y_e,'ro')
#plt.plot(X_plot,f_c,'k-')
ax.plot(X_plot,f_e,'r-')
ax.plot(X_test,Y_test,'g-')
#plt.plot(X_test,Y_sese,'b-.')
#plt.plot(X_test,yy_test,'b-')

plt.plot(X_test,Y_test_up,'b--')
plt.plot(X_test,Y_test_down,'b--')
#ax.fill_between(X_test, Y_test_down, Y_test_up)
#plt.plot(X2, mu[1:], 'bx')
plt.show()
