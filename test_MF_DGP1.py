
import numpy as np
import GPy
import seaborn as sns
from matplotlib import pyplot as plt


def f_low(x,w):
    #return np.sin(8*np.pi*x)
    #return x**2
    #return np.cos(15.*x)
    #w = np.random.normal(0,10,5)
    tmp = 0
    for i in range(len(w)):
        tmp = tmp+np.cos(w[i]*x)/len(w)
    return tmp

def f_high(x,w):
    #return (x-np.sqrt(2)) * (f_low(x))**2
    #return np.sin(2*np.pi*x**2)
    #return x*np.exp(np.cos(30*x-3))-1
    #w = np.random.normal(0,10,5)
    tmp = 0
    for i in range(len(w)-1):
        tmp = tmp + np.cos(w[i]*x)/(len(w)-1)
    tmp = tmp * np.cos(w[-1]*x)
    return tmp

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
    K_inv = np.linalg.inv(K_ee+0.0001*np.eye(K_ee.shape[0]))
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

    ax = np.linspace(0,1,100)
    bx = -2 + ax * 3.
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

def heat_map(num_cheap, num_sample):

    np.random.seed(59)
    X1 = np.random.rand(num_cheap)[:,None]
    Y1 = f_low(X1)

    var = 0.7
    vvaar1 = 0.6
    lenl1 = 0.1

    K1 = GPy.kern.RBF(input_dim = 1, variance = vvaar1, lengthscale = lenl1)
    m1 = GPy.models.GPRegression(X1, Y1, K1, noise_var = 0.0001)
    m1.optimize(messages=True)
    m1.optimize_restarts(num_restarts = 10)

    X2 = np.linspace(0,1,num_sample)[:,None]

    mu, v = m1.predict(X2, full_cov = True)

    k_eff1 = Cov_matrix_eff(mu, v, 1, 0.5)

    K_test = GPy.kern.RBF(1)
    k_rbf = K_test.K(X2)

    k_eff2 = np.multiply(k_rbf, k_eff1)

    #p1 = sns.heatmap(k_rbf)

    plt.figure(1)

    #plt.subplot(1,2,1)
    #plt.imshow(k_eff1, cmap='hot', interpolation='nearest')

    #plt.subplot(1,2,2)
    #plt.imshow(k_eff2, cmap='hot', interpolation='nearest')

    ax = sns.heatmap(k_eff2, linewidth=0.5)

    plt.figure(2)
    plt.plot(X2, np.random.multivariate_normal(np.zeros(len(X2)), k_eff2, 1).T, 'r-')
    plt.plot(X2, np.random.multivariate_normal(np.zeros(len(X2)), k_eff2, 1).T, 'b-')
    plt.plot(X1, Y1, 'kx')

    plt.show()


def model_case_1(num_cheap, num_expensive, idx, pathology, noise_level):

    var = 0.7
    vvaar1 = 0.6
    lenl1 = 0.1

    w1 = np.random.normal(0,40,15).reshape(-1,1)
    w2 = np.random.normal(10,1,1).reshape(-1,1)
    w3 = np.vstack((w1,w2))

    #np.random.seed(59)
    #X1 = np.random.rand(num_cheap)[:,None]
    X1 = np.linspace(0,1,num_cheap)[:,None]
    #X2 = np.random.rand(num_expensive)[:,None]
    X2 = np.linspace(0,1,num_expensive)[:,None]

    X_plot = np.linspace(0,1,200)[:,None]

    f_c=f_low(X_plot,w1)
    f_e=f_high(X_plot,w3)

    # idx = 1 for high-low fidelity
    # idx = 2 for high-low noise

    if idx is 1:
        Yc = f_low(X1,w1)
    else:
        Yc = f_high(X1,w3)+ noise_level * np.random.normal(0,1,num_cheap)[:,None]

    Ye = f_high(X2,w3) + 0.001 * np.random.normal(0,1,num_expensive)[:,None]

    #K1 = GPy.kern.RBF(1)
    K1 = GPy.kern.RBF(input_dim = 1, variance = vvaar1, lengthscale = lenl1)
    m1 = GPy.models.GPRegression(X1, Yc, K1, noise_var = 0.0001)
    m1.optimize(messages=True)
    m1.optimize_restarts(num_restarts = 10)
    m1.plot()
    plt.plot(X_plot,f_c,'k--')

    #Ka = GPy.kern.RBF(input_dim = 1, variance = vvaar1, lengthscale = lenl1)
    #Xa = np.vstack((X1,X2))
    #Ya = np.vstack((Yc,Ye))
    #ma = GPy.models.GPRegression(Xa, Ya, Ka, noise_var = 0.0001)
    #ma.optimize(messages=True)
    #ma.optimize_restarts(num_restarts = 10)

    Kb = GPy.kern.RBF(input_dim = 1, variance = 0.5, lengthscale = 0.3)
    mb = GPy.models.GPRegression(X2, Ye, Kb, noise_var = 0.0001)
    mb.optimize(messages=True)
    mb.optimize_restarts(num_restarts = 10)

    mu, v = m1.predict(X2, full_cov = True)

    ax = np.linspace(0,1,100)
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

    print(fav)

    X_test = np.linspace(0,1,99)[:,None]

    Y_test=[]
    Y_test_up=[]
    Y_test_down=[]

    #K_test1 = GPy.kern.RBF(1)

    for xx in X_test:
        Xnew = np.vstack((xx,X2))
        mu, v = m1.predict(Xnew, full_cov = True)
        #print(mu)
        MM = Cov_matrix_eff(mu, v, fav[0], fav[1])

        if pathology:
            kk_test = K_test.K(Xnew)
            GG = np.multiply(kk_test, MM)
        else:
            GG = MM

        pred_mu, pred_v = predict_eff(GG, Ye)

        Y_test.append(pred_mu)
        Y_test_up.append(pred_mu+1.96*pred_v)
        Y_test_down.append(pred_mu-1.96*pred_v)


    if idx is 2:

        figa = ma.plot()
        plt.plot(X2,Ye,'ro')

        plt.plot(X_plot,f_e,'r-')
        plt.plot(X_test,Y_test,'k-')

        plt.plot(X_test,Y_test_up,'b--')
        plt.plot(X_test,Y_test_down,'b--')

        #plt.figure(2)
        #figb = mb.plot()

    else:

        #plt.plot(X1,Yc,'kx')
        plt.figure(2)
        plt.plot(X2,Ye,'ro')

        plt.plot(X_plot,f_e,'r-')
        plt.plot(X_test,Y_test,'k-')

        #plt.plot(X_test,Y_test_up,'b--')
        #plt.plot(X_test,Y_test_down,'b--')
        plt.fill_between(np.array(X_test).flatten(),np.array(Y_test_up).flatten(),np.array(Y_test_down).flatten(),facecolor='g',alpha=0.3)
        #plt.figure(3)
        figb = mb.plot()
        plt.plot(X_plot,f_e,'r-')
        plt.xlim(-0.1,1.1)

    plt.show()


    return 0

#model_case_1(40, 10, 2, 1, 0.25)
model_case_1(30, 10, 1, 1, 0.05)
#model_case_1(30, 10, 1, 0, 0.05)
#heat_map(40,30)
