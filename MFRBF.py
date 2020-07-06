import numpy as np
from GPy.kern import Kern
from GPy import Param, Model

class MFRBF(Kern):

    def __init__(self,input_dim,variance=1.,lengthscale=1.,mu=None,v=None,active_dims=None):
        super(MFRBF, self).__init__(input_dim, active_dims, 'MFRBF')
        self.mu = mu
        self.v = v
        self.variance = Param('variance', variance)
        self.lengthscale = Param('lengtscale', lengthscale)
        self.link_parameters(self.variance, self.lengthscale)

    def parameters_changed(self):
        # nothing todo here
        pass

    def K(self,X,X2):
        if X2 is None: X2 = X

        m_eff = np.zeros((len(self.mu),len(self.mu)))
        for i in range(len(self.mu)):

            for j in range(len(self.mu)):

                if i == j:
                    m_eff[i,j] = self.variance

                else:
                    if m_eff[j,i] == 0:
                        d_mu = self.mu[i]-self.mu[j]
                        k_ii = self.v[i,i]
                        k_jj = self.v[j,j]
                        k_ij = self.v[i,j]
                        tmp_1 = 1 + (k_ii+k_jj-2*k_ij)/self.lengthscale**2
                        tmp_exp = np.exp((-1.)*d_mu**2/2/self.lengthscale**2/tmp_1)
                        tmp_sqrt = tmp_1**(-1/2)
                        m_eff[i,j] = self.variance * tmp_exp * tmp_sqrt
                    else:
                        m_eff[i,j] = m_eff[j,i]

        return m_eff+0.0001*np.eye(m_eff.shape[0])

    def Kdiag(self,X):
        return self.variance*np.ones(len(mu))

    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: X2 = X

        k = self.K(X)

        dvar = k/self.variance

        # Calculate the gradient component with respect to lengthscale
        m_eff = np.zeros((len(self.mu),len(self.mu)))

        for i in range(len(self.mu)):
            for j in range(len(self.mu)):

                if i == j:
                    m_eff[i,j] = 0.
                else:
                    if m_eff[j,i] == 0:
                        d_mu = self.mu[i]-self.mu[j]
                        k_ii = self.v[i,i]
                        k_jj = self.v[j,j]
                        k_ij = self.v[i,j]
                        tmp1 =  k_ii + k_jj - 2*k_ij
                        tmp2 = tmp1 + self.lengthscale**2
                        m_eff[i,j] = tmp1/tmp2/self.lengthscale + self.lengthscale * d_mu**2/tmp2**2
                    else:
                        m_eff[i,j] = m_eff[j,i]

        dl = np.multiply(m_eff, k)

        self.variance.gradient = np.sum(dvar*dL_dK)
        self.lengthscale.gradient = np.sum(dl*dL_dK)

    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance.gradient = np.sum(dL_dKdiag)
        # here self.lengthscale and self.power have no influence on Kdiag so target[1:] are unchanged

    def gradients_X(self,dL_dK,X,X2):
        """derivative of the covariance matrix with respect to X."""
        if X2 is None: X2 = X
        pass

    def gradients_X_diag(self,dL_dKdiag,X):
        # no diagonal gradients
        pass



class MFCosine(Kern):

    def __init__(self,input_dim,variance=1.,lengthscale=1.,mu=None,v=None,active_dims=None):
        super(MFCosine, self).__init__(input_dim, active_dims, 'MFCosine')
        #assert input_dim == 1, "For this kernel we assume input_dim=1"
        self.mu = mu
        self.v = v
        self.variance = Param('variance', variance)
        self.lengthscale = Param('lengtscale', lengthscale)
        self.link_parameters(self.variance, self.lengthscale)

    def parameters_changed(self):
        # nothing todo here
        pass

    def K(self,X,X2):
        if X2 is None: X2 = X

        m_eff = np.zeros((len(self.mu),len(self.mu)))
        for i in range(len(self.mu)):
            for j in range(len(self.mu)):
                if i == j:
                    m_eff[i,j] = self.variance

                else:
                    if m_eff[j,i] == 0:
                        d_mu = self.mu[i]-self.mu[j]
                        k_ii = self.v[i,i]
                        k_jj = self.v[j,j]
                        k_ij = self.v[i,j]
                        tmp_1 = (k_ii+k_jj-2*k_ij)/2/self.lengthscale**2
                        tmp_exp = np.exp((-1.)*tmp_1)
                        tmp_cosine = np.cos(d_mu/self.lengthscale)
                        m_eff[i,j] = 0.5*self.variance*(1+tmp_exp*tmp_cosine)
                    else:
                        m_eff[i,j] = m_eff[j,i]

        return m_eff+0.0001*np.eye(m_eff.shape[0])

    def Kdiag(self,X):
        return self.variance*np.ones(len(mu))

    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: X2 = X

        k = self.K(X)
        dvar = k/self.variance

        # Calculate the gradient component with respect to lengthscale
        m_eff = np.zeros((len(self.mu),len(self.mu)))

        for i in range(len(self.mu)):
            for j in range(len(self.mu)):

                if i == j:
                    m_eff[i,j] = 0.

                else:
                    if m_eff[j,i] == 0:
                        d_mu = self.mu[i]-self.mu[j]
                        k_ii = self.v[i,i]
                        k_jj = self.v[j,j]
                        k_ij = self.v[i,j]
                        tmp = 2*k_ij-k_ii-k_jj
                        tmp1 = tmp/self.lengthscale**3
                        tmp2 = d_mu/self.lengthscale**2
                        tmp3 = np.sin(d_mu/self.lengthscale)*tmp2 + np.cos(d_mu/self.lengthscale)*tmp1
                        m_eff[i,j] = 0.5*self.variance*np.exp(tmp/2/self.lengthscale**2)*tmp3
                    else:
                        m_eff[i,j] = m_eff[j,i]

        dl = m_eff

        self.variance.gradient = np.sum(dvar*dL_dK)
        self.lengthscale.gradient = np.sum(dl*dL_dK)

    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance.gradient = np.sum(dL_dKdiag)
        # here self.lengthscale and self.power have no influence on Kdiag so target[1:] are unchanged

    def gradients_X(self,dL_dK,X,X2):
        """derivative of the covariance matrix with respect to X."""
        if X2 is None: X2 = X
        pass

    def gradients_X_diag(self,dL_dKdiag,X):
        # no diagonal gradients
        pass


minor update
