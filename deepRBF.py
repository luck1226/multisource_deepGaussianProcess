
import numpy as np
from scipy.special import erf
from GPy.kern import Kern
from GPy import Param, Model

class deepRBF(Kern):

    def __init__(self,input_dim,variance1=1.,lengthscale=1.,variance2=1.,active_dims=None):
        super(deepRBF, self).__init__(input_dim, active_dims, 'deep_rbf')
        self.variance1 = Param('variance1', variance1)
        self.lengthscale = Param('lengtscale', lengthscale)
        self.variance2 = Param('variance2', variance2)
        self.link_parameters(self.variance1, self.lengthscale, self.variance2)

    def parameters_changed(self):
        pass

    def K(self,X,X2):
        if X2 is None: X2 = X
        dist2 = (np.square(X[:,np.newaxis]-X2).sum(axis=2))/self.lengthscale**2
        tmp1 = 1.-np.exp((-1.)*dist2/2.)
        tmp2 = 2.*self.variance1*tmp1
        return self.variance2*np.power(1 + tmp2,-0.5)

    def Kdiag(self,X):
        return self.variance2*np.ones(X.shape[0])

    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: X2 = X

        dist2 = (np.square(X[:,np.newaxis]-X2).sum(axis=2))/self.lengthscale**2
        tmp1 = 1.-np.exp((-1.)*dist2/2.)
        tmp2 = 2.*self.variance1*tmp1

        dvar2 = np.power(1 + tmp2, -0.5)
        dvar1 = (-1.)*self.variance2*tmp1*np.power(1+tmp2,-1.5)
        dl = self.variance1*self.variance2*np.exp((-1.)*dist2)*dist2/self.lengthscale*np.power(1+tmp2,-1.5)

        self.variance1.gradient = np.sum(dvar1*dL_dK)
        self.lengthscale.gradient = np.sum(dl*dL_dK)
        self.variance2.gradient = np.sum(dvar2*dL_dK)

    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance2.gradient = np.sum(dL_dKdiag)

    def gradients_X(self,dL_dK,X,X2):
        """derivative of the covariance matrix with respect to X."""
        if X2 is None: X2 = X
        dist2 = (np.square(X[:,np.newaxis]-X2).sum(axis=2))/self.lengthscale**2
        tmp1 = 1.-np.exp((-1.)*dist2/2.)
        tmp2 = 2.*self.variance1*tmp1
        tmp3 = np.power(1+tmp2,-1.5)

        dX_tmp = (-1.)*self.variance1*self.variance2/self.lengthscale**2 * np.exp((-1.)*dist2/2) * tmp3
        return ((dL_dK*dX_tmp)[:,:,None]*(X[:,None,:] - X2[None,:,:])).sum(1)

    def gradients_X_diag(self,dL_dKdiag,X):
        # no diagonal gradients
        pass


class deepCosine(Kern):

    def __init__(self,input_dim,variance1=1.,lengthscale=1.,variance2=1.,active_dims=None):
        super(deepCosine, self).__init__(input_dim, active_dims, 'deep_cosine')
        #assert input_dim == 1, "For this kernel we assume input_dim=1"
        self.variance1 = Param('variance1', variance1)
        self.lengthscale = Param('lengtscale', lengthscale)
        self.variance2 = Param('variance2', variance2)
        #self.lengthscale2 = Param('lengthscale2', lengthscale2)
        self.link_parameters(self.variance1, self.lengthscale, self.variance2)

    def parameters_changed(self):
        # nothing todo here
        pass

    def K(self,X,X2):
        if X2 is None: X2 = X
        dist2 = (np.square(X[:,np.newaxis]-X2).sum(axis=2))/self.lengthscale**2
        tmp1 = np.exp((-1.)*dist2/2.)-1.
        tmp2 = self.variance1*tmp1
        return 0.5*self.variance2*(1+np.exp(tmp2))

    def Kdiag(self,X):
        return self.variance2*np.ones(X.shape[0])


    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: X2 = X

        dist2 = (np.square(X[:,np.newaxis]-X2).sum(axis=2))/self.lengthscale**2
        tmp1 = np.exp((-1.)*dist2/2.)-1.
        tmp2 = self.variance1*tmp1
        tmp3 = np.exp(tmp2)

        dvar2 = 0.5*(1+np.exp(tmp2))
        #dvar1 = (-1.)*self.variance2*tmp1*np.power(1+tmp2,-1.5)
        dvar1 = 0.5*self.variance2*tmp3*tmp1
        #dl = self.variance1*self.variance2*np.exp((-1.)*dist2)*dist2/self.lengthscale*np.power(1+tmp2,-1.5)
        dl = self.variance1*self.variance2*tmp3*(tmp1+1)*dist2/self.lengthscale

        self.variance1.gradient = np.sum(dvar1*dL_dK)
        self.lengthscale.gradient = np.sum(dl*dL_dK)
        self.variance2.gradient = np.sum(dvar2*dL_dK)

    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance2.gradient = np.sum(dL_dKdiag)
        # here self.lengthscale and self.power have no influence on Kdiag so target[1:] are unchanged

    def gradients_X(self,dL_dK,X,X2):
        """derivative of the covariance matrix with respect to X."""
        pass
        #if X2 is None: X2 = X
        #dist2 = (np.square(X[:,np.newaxis]-X2).sum(axis=2))/self.lengthscale**2
        #tmp1 = 1.-np.exp((-1.)*dist2/2.)
        #tmp2 = 2.*self.variance1*tmp1
        #tmp3 = np.power(1+tmp2,-1.5)

        #dX_tmp = (-1.)*self.variance1*self.variance2/self.lengthscale**2 * np.exp((-1.)*dist2/2) * tmp3
        #return ((dL_dK*dX_tmp)[:,:,None]*(X[:,None,:] - X2[None,:,:])).sum(1)

    def gradients_X_diag(self,dL_dKdiag,X):
        # no diagonal gradients
        pass


class deep3RBF(Kern):

    def __init__(self,input_dim,variance1=1.,lengthscale=1.,variance2=1.,variance3=1.,active_dims=None):
        super(deep3RBF, self).__init__(input_dim, active_dims, 'deep3rbf')
        #assert input_dim == 1, "For this kernel we assume input_dim=1"
        self.variance1 = Param('variance1', variance1)
        self.lengthscale = Param('lengtscale', lengthscale)
        self.variance2 = Param('variance2', variance2)
        self.variance3 = Param('variance3',variance3)
        self.link_parameters(self.variance1, self.lengthscale, self.variance2, self.variance3)

    def parameters_changed(self):
        # nothing todo here
        pass

    def K(self,X,X2):
        if X2 is None: X2 = X
        dist2 = (np.square(X[:,np.newaxis]-X2).sum(axis=2))/self.lengthscale**2
        tmp1 = 2.*(1.-np.exp((-1.)*dist2/2.))+0.000001
        tmp2 = 1./np.sqrt(tmp1)/self.variance1
        tmp3 = erf(tmp2)
        return self.variance3/np.sqrt(1+2.*self.variance2)*(1-tmp3)+self.variance3*tmp3

    def Kdiag(self,X):
        return self.variance3*np.ones(X.shape[0])

    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: X2 = X

        dist2 = (np.square(X[:,np.newaxis]-X2).sum(axis=2))/self.lengthscale**2
        tmp1 = 2.*(1.-np.exp((-1.)*dist2/2.))+0.000001
        tmp2 = 1./np.sqrt(tmp1)/self.variance1
        tmp3 = erf(tmp2)

        dvar3 = 1./np.sqrt(1+2.*self.variance2)*(1-tmp3)+tmp3
        dvar2 = 2.*self.variance3/(1+2.*self.variance2)**(1.5)*(1-tmp3)

        dvar1 = 2.*self.variance3/self.variance1/np.sqrt(np.pi)*np.exp((-1.)*tmp2**2)*tmp2*(-1.+1/np.sqrt(1+2.*self.variance2))

        dl_1 = self.variance3*self.variance1**2/self.lengthscale*(1-1/np.sqrt(1+2*self.variance2))/np.sqrt(np.pi)
        dl_2 = np.exp(-1.*tmp2**2)*tmp2**3*np.exp((-0.5)*dist2)*dist2
        dl = dl_1*dl_2

        self.variance1.gradient = np.sum(dvar1*dL_dK)
        self.lengthscale.gradient = np.sum(dl*dL_dK)
        self.variance2.gradient = np.sum(dvar2*dL_dK)
        self.variance3.gradient = np.sum(dvar3*dL_dK)

    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance3.gradient = np.sum(dL_dKdiag)
        # here self.lengthscale and self.power have no influence on Kdiag so target[1:] are unchanged

    def gradients_X(self,dL_dK,X,X2):
        """derivative of the covariance matrix with respect to X."""
        pass

    def gradients_X_diag(self,dL_dKdiag,X):
        # no diagonal gradients
        pass
