import numpy as np
import matplotlib.pyplot as plt

import GPy
import emukit.multi_fidelity
import emukit.test_functions
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays

from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels, NonLinearMultiFidelityModel

high_fidelity = emukit.test_functions.non_linear_sin.nonlinear_sin_high
low_fidelity = emukit.test_functions.non_linear_sin.nonlinear_sin_low

np.random.seed(59)

#num_cheap = 30
#num_expensive = 10

x_plot = np.linspace(0, 1, 200)[:, None]

y_plot_l = low_fidelity(x_plot)
y_plot_h = high_fidelity(x_plot)

n_low_fidelity_points = 30
n_high_fidelity_points = 10

x_train_l = np.random.rand(n_low_fidelity_points)[:, None]
y_train_l = low_fidelity(x_train_l)

x_train_h = np.random.rand(n_high_fidelity_points)[:, None]

x_train_h[0]=0.43424816
x_train_h[1]=0.64439393
x_train_h[2]=0.96176674
x_train_h[3]=0.43949222
x_train_h[4]=0.4391256
x_train_h[5]=0.64022806
x_train_h[6]=0.9130996
x_train_h[7]=0.55606022
x_train_h[8]=0.53545111
x_train_h[9]=0.44325942

y_train_h = high_fidelity(x_train_h)

### Convert lists of arrays to ND-arrays augmented with fidelity indicators

X_train, Y_train = convert_xy_lists_to_arrays([x_train_l, x_train_h], [y_train_l, y_train_h])

X_plot = convert_x_list_to_array([x_plot, x_plot])
X_plot_low = X_plot[:200]
X_plot_high = X_plot[200:]


#X1 = np.random.rand(num_cheap)[:,None]
#X2 = np.random.rand(num_expensive)[:,None]
#X_test = np.linspace(0,1,99)[:,None]

#y_plot_1 = low_fidelity(X_test)
#y_plot_2 = high_fidelity(X_test)

#y1 = low_fidelity(X1)
#y2 = high_fidelity(X2)

base_kernel = GPy.kern.RBF
kernels = make_non_linear_kernels(base_kernel, 2, X_train.shape[1] - 1)
nonlin_mf_model = NonLinearMultiFidelityModel(X_train, Y_train, n_fidelities=2, kernels=kernels,
                                              verbose=True, optimization_restarts=5)
for m in nonlin_mf_model.models:
    m.Gaussian_noise.variance.fix(0)

nonlin_mf_model.optimize()


hf_mean_nonlin_mf_model, hf_var_nonlin_mf_model = nonlin_mf_model.predict(X_plot_high)
hf_std_nonlin_mf_model = np.sqrt(hf_var_nonlin_mf_model)

lf_mean_nonlin_mf_model, lf_var_nonlin_mf_model = nonlin_mf_model.predict(X_plot_low)
lf_std_nonlin_mf_model = np.sqrt(lf_var_nonlin_mf_model)


## Plot posterior mean and variance of nonlinear multi-fidelity model

#plt.figure(figsize=(12,8))
plt.fill_between(x_plot.flatten(), (lf_mean_nonlin_mf_model - 1.96*lf_std_nonlin_mf_model).flatten(),
                 (lf_mean_nonlin_mf_model + 1.96*lf_std_nonlin_mf_model).flatten(), color='g', alpha=0.3)
plt.fill_between(x_plot.flatten(), (hf_mean_nonlin_mf_model - 1.96*hf_std_nonlin_mf_model).flatten(),
                 (hf_mean_nonlin_mf_model + 1.96*hf_std_nonlin_mf_model).flatten(), color='y', alpha=0.3)
plt.plot(x_plot, y_plot_l, 'b')
plt.plot(x_plot, y_plot_h, 'r')
plt.plot(x_plot, lf_mean_nonlin_mf_model, '--', color='g')
plt.plot(x_plot, hf_mean_nonlin_mf_model, '--', color='y')
plt.scatter(x_train_h, y_train_h, color='r')
#plt.xlabel('x')
#plt.ylabel('f (x)')
#plt.xlim(0, 1)
#plt.legend(['Low Fidelity', 'High Fidelity', 'Predicted Low Fidelity', 'Predicted High Fidelity'])
#plt.title('Nonlinear multi-fidelity model fit to low and high fidelity functions');

#plt.plot(X_test,y_plot_1,'b')
#plt.plot(X_test,y_plot_2,'r')
#print(x_train_h)

plt.show()
