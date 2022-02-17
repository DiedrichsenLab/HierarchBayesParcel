from arrangements import ArrangeIndependent
from emissions import MixGaussian, MixGaussianExp, MixVMF, MixGaussianGamma, mean_adjusted_sse
import os # to handle path information
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import nibabel as nb
from nilearn import plotting

class FullModel:
    def __init__(self, arrange,emission):
        self.arrange = arrange
        self.emission = emission
        self.nparams = self.arrange.nparams + self.emission.nparams

    def sample(self, num_subj=10):
        U = self.arrange.sample(num_subj)
        Y = self.emission.sample(U)
        return U, Y

    def Estep(self,Y=None, separate_ll=False):
        if Y is not None: 
            self.emission.initialize(Y)
        emloglik = self.emission.Estep()
        Uhat, ll_a = self.arrange.Estep(emloglik)
        ll_e = (Uhat * emloglik).sum()
        if separate_ll: 
            return Uhat,[ll_e,ll_a.sum()]
        else: 
            return Uhat,ll_a.sum()+ll_e

    def fit_em(self, Y, iter=30, tol=0.01, seperate_ll=False,
               fit_emission=True, fit_arrangement=True):
        """ Run the EM-algorithm on a full model
        this demands that both the Emission and Arrangement model
        have a full Estep and Mstep and can calculate the likelihood, including the partition function

        Args:
            Y (3d-ndarray): numsubj x N x numvoxel array of data
            iter (int): Maximal number of iterations (def:30)
            tol (double): Tolerance on overall likelihood (def: 0.01)
            seperate_ll (bool): Return arrangement and emission LL separetely
        Returns:
            model (Full Model): fitted model (also updated)
            ll (ndarray): Log-likelihood of full model as function of iteration
                If seperate_ll, the first column is ll_A, the second ll_E
            theta (ndarray): History of the parameter vector

        """
        # Initialize the tracking
        ll = np.zeros((iter, 2))
        theta = np.zeros((iter, self.emission.nparams+self.arrange.nparams))
        self.emission.initialize(Y)
        for i in range(iter):
            # Track the parameters
            theta[i, :] = np.concatenate([self.emission.get_params(), self.arrange.get_params()])

            # Get the (approximate) posterior p(U|Y)
            emloglik = self.emission.Estep()
            Uhat, ll_A = self.arrange.Estep(emloglik)
            # Compute the expected complete logliklihood
            ll_E = pt.sum(Uhat * emloglik, dim=(1, 2))
            ll[i, 0] = pt.sum(ll_A)
            ll[i, 1] = pt.sum(ll_E)
            # Check convergence
            if i == iter-1 or ((i > 1) and (np.abs(ll[i,:].sum() - ll[i-1,:].sum()) < tol)):
                break

            # Updates the parameters
            if fit_emission:
                self.emission.Mstep(Uhat)
            if fit_arrangement:
                self.arrange.Mstep()

        if seperate_ll:
            return self, ll, theta, Uhat
        else:
            return self, ll.sum(axis=1), theta, Uhat

    def fit_sml(self, Y, iter=60, stepsize= 0.8, seperate_ll=False, estep='sample'):
        """ Runs a Stochastic Maximum likelihood algorithm on a full model.
        The emission model is still assumed to have E-step and Mstep.
        The arrangement model is has a postive and negative phase estep,
        and a gradient M-step. The arrangement likelihood is not necessarily
        FUTURE EXTENSIONS:
        * Sampling of subjects from training set
        * initialization of parameters
        * adaptitive stopping criteria
        * Adaptive stepsizes
        * Gradient acceleration methods
        Args:
            Y (3d-ndarray): numsubj x N x numvoxel array of data
            iter (int): Maximal number of iterations
            stepsize (double): Fixed step size for MStep
        Returns:
            model (Full Model): fitted model (also updated)
            ll (ndarray): Log-likelihood of full model as function of iteration
                If seperate_ll, the first column is ll_A, the second ll_E
            theta (ndarray): History of the parameter vector
        """
        # Initialize the tracking
        ll = np.zeros((iter,2))
        theta = np.zeros((iter, self.emission.nparams+self.arrange.nparams))
        self.emission.initialize(Y)
        for i in range(iter):
            # Track the parameters
            theta[i, :] = np.concatenate([self.arrange.get_params(),self.emission.get_params()])

            # Get the (approximate) posterior p(U|Y)
            emloglik = self.emission.Estep()
            if estep=='sample':
                Uhat,ll_A = self.arrange.epos_sample(emloglik,num_chains=self.arrange.epos_numchains)
                self.arrange.eneg_sample(num_chains=self.arrange.eneg_numchains)
            elif estep=='ssa':
                Uhat,ll_A = self.arrange.epos_ssa(emloglik)
                self.arrange.eneg_ssa()

            # Compute the expected complete logliklihood
            ll_E = np.sum(Uhat * emloglik,axis=(1,2))
            ll[i,0]=np.sum(ll_A)
            ll[i,1]=np.sum(ll_E)

            # Run the Mstep
            self.emission.Mstep(Uhat)
            self.arrange.Mstep(stepsize)

        if seperate_ll:
            return self,ll[:i+1,:], theta[:i+1,:]
        else:
            return self,ll[:i+1,:].sum(axis=1), theta[:i+1,:]

    def ELBO(self,Y):
        """Evidence lower bound of the data under the full model
        Args:
            Y (nd-array): numsubj x N x P array of data
        Returns:
            ELBO (nd-array): Evidence lower bound - should be relatively tight
            Uhat (nd-array): numsubj x K x P array of expectations
            ll_E (nd-array): emission logliklihood of data (numsubj,)
            ll_A (nd-array): arrangement logliklihood of data (numsubj,)
            lq (nd-array): <log q(u)> under q: Entropy
        """
        self.emission.initialize(Y)
        emloglik=self.emission.Estep()
        try:
            Uhat,ll_A,QQ = self.arrange.Estep(emloglik,return_joint=True)
            lq = np.sum(np.log(QQ)*QQ,axis=(1,2))
        except:
            # Assume independence:
            Uhat,ll_A = self.arrange.Estep(emloglik)
            lq = np.sum(np.log(Uhat)*Uhat,axis=(1,2))
            # This is the same as:
            # Uhat2 = Uhat[0,:,0]*Uhat[0,:,1].reshape(-1,1)
            # l_test = np.sum(np.log(Uhat2)*Uhat2)
        ll_E = np.sum(emloglik*Uhat,axis=(1,2))
        ELBO = ll_E + ll_A - lq
        return  ELBO, Uhat, ll_E,ll_A,lq


    def get_params(self):
        """Get the concatenated parameters from arrangemenet + emission model
        Returns:
            theta (ndarrap)
        """
        return np.concatenate([self.arrange.get_params(),self.emission.get_params()])

    def get_param_indices(self,name):
        """Return the indices for the full model theta vector

        Args:
            name (str): Parameter name in the format of 'arrange.logpi'
                        or 'emission.V'
        Returns:
            indices (np.ndarray): 1-d numpy array of indices into the theta vector
        """
        names = name.split(".")
        if len(names)==2:
            ind=vars(self)[names[0]].get_param_indices(names[1])
            if names[0]=='emission':
                ind=ind+self.arrange.nparams
            return ind
        else:
            raise NameError('Parameter name needs to be model.param')


def _plot_loglike(loglike, true_loglike, color='b'):
    plt.figure()
    plt.plot(loglike, color=color)
    plt.axhline(y=true_loglike, color='r', linestyle=':')


def _plot_diff(true_param, predicted_params, index=None, name='V'):
    """ Plot the model parameters differences.

    Args:
        true_param: the params from the true model
        predicted_params: the estimated params
        index: the matching index of parameters
        color: the line color for the differences
    Returns: a matplotlib object to be plot
    """
    # Convert input to tensor if ndarray
    if type(true_param) is np.ndarray:
        true_param = pt.tensor(true_param, dtype=pt.get_default_dtype())
    if type(predicted_params) is np.ndarray:
        predicted_params = pt.tensor(predicted_params, dtype=pt.get_default_dtype())

    N, K = true_param.shape
    diff = pt.empty(predicted_params.shape[0], K)

    for i in range(predicted_params.shape[0]):
        this_pred = predicted_params[i].reshape(N, K)
        for k in range(K):
            if index is not None:
                dist = pt.linalg.norm(this_pred[:, k] - true_param[:, index[i, k]])
            else:
                dist = pt.linalg.norm(this_pred[:, k] - true_param[:, k])
            diff[i, k] = dist
    plt.figure()
    plt.plot(diff)
    plt.title('the differences between true %ss and estimated %ss for each k' % (name, name))


def _plt_single_param_diff(theta_true, theta, name=None):
    plt.figure()
    if name is not None:
        plt.title('True %s (red) vs estimated %s (blue)' % (name, name))

    iter = theta.shape[0]
    theta_true = np.repeat(theta_true, iter)
    plt.plot(theta_true, linestyle='--', color='r')
    plt.plot(theta, color='b')


def generate_data(emission, k=2, dim=3, p=1000,
                  num_sub=10, beta=1, alpha=1, signal_type=0):
    model_name = ["GMM", "GMM_exp", "GMM_gamma", "VMF"]
    arrangeT = ArrangeIndependent(K=k, P=p, spatial_specific=False, remove_redundancy=False)
    U = arrangeT.sample(num_subj=num_sub)
    if signal_type == 0:
        signal = np.random.exponential(beta, (num_sub, p))
    elif signal_type == 1:
        signal = np.random.gamma(alpha, beta, (num_sub, p))
    else:
        raise ValueError("The value of signal strength must satisfy a distribution, 0 - exponential; 1 - gamma.")

    if emission == 0:  # GMM
        emissionT = MixGaussian(K=k, N=dim, P=p)
    elif emission == 1:  # GMM with exponential signal strength
        emissionT = MixGaussianExp(K=k, N=dim, P=p)
    elif emission == 2:  # GMM with gamma signal strength
        emissionT = MixGaussianGamma(K=k, N=dim, P=p)
    elif emission == 3:
        emissionT = MixVMF(K=k, N=dim, P=p)
    else:
        raise ValueError("The value of emission must be 0(GMM), 1(GMM_exp), 2(GMM_gamma), or 3(VMF).")

    if (emission == 1) or (emission == 2):
        data = emissionT.sample(U, signal)
    elif emission == 3:
        data = emissionT.sample(U)
        signal = np.repeat(signal[:, np.newaxis, :], dim, axis=1)
        data = data * signal
    else:
        data = emissionT.sample(U)

    return data, U


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def matching_params(true_param, predicted_params, once=False):
    """ Matching the estimated parameters with the true one, return indices

    Args:
        true_param: the true parameter, shape (N, K)
        predicted_params: the estimated parameter. Shape (iter, N*K)
        once: True - perform matching in every iteration. Otherwise only take
              matching once using the estimated param of first iteration

    Returns: The matching index

    """
    # Convert input to tensor if ndarray
    if type(true_param) is np.ndarray:
        true_param = pt.tensor(true_param, dtype=pt.get_default_dtype())
    if type(predicted_params) is np.ndarray:
        predicted_params = pt.tensor(predicted_params, dtype=pt.get_default_dtype())

    N, K = true_param.shape
    if once:
        index = pt.empty(K, )
        for k in range(K):
            tmp = pt.linalg.norm(predicted_params[0].reshape(N, K)[:, k] - true_param.transpose(0, 1), dim=1)
            index[k] = pt.argmin(tmp)
            true_param[:, pt.argmin(tmp)] = pt.tensor(float('inf'))
        index.expand(predicted_params.shape[0], K)
    else:
        index = pt.empty(predicted_params.shape[0], K)
        for i in range(index.shape[0]):
            this_pred = predicted_params[i].reshape(N, K)
            this_true = pt.clone(true_param).transpose(0, 1)
            for k in range(K):
                tmp = pt.linalg.norm(this_pred[:, k] - this_true, dim=1)
                index[i, k] = pt.argmin(tmp)
                this_true[pt.argmin(tmp), :] = pt.tensor(float('inf'))

    return index.int()


def _simulate_full_GMM(K=5, P=100, N=40, num_sub=10, max_iter=50):
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionT = MixGaussian(K=K, N=N, P=P)
    # emissionT.random_params()

    # Step 2: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=num_sub)
    Y = emissionT.sample(U)

    # Step 3: Compute the log likelihood from the true model
    theta_true = np.concatenate([emissionT.get_params(), arrangeT.get_params()])
    T = FullModel(arrangeT, emissionT)
    Uhat, loglike_true = T.Estep(Y=Y)  # Run only Estep!

    # Step 4: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionM = MixGaussian(K=K, N=N, P=P)

    # Step 5: Estimate the parameter thetas to fit the new model using EM
    M = FullModel(arrangeM, emissionM)
    M, ll, theta, _ = M.fit_em(Y=Y, iter=max_iter, tol=0.00001)

    # Plot fitting results
    _plot_loglike(np.trim_zeros(ll, 'b'), loglike_true, color='b')
    true_V = theta_true[emissionT.get_param_indices('V')].reshape(N, K)
    predicted_V = theta[:, M.emission.get_param_indices('V')]
    idx = matching_params(true_V, predicted_V, once=False)

    _plot_diff(true_V, predicted_V, index=idx, name='V')
    _plt_single_param_diff(theta_true[M.emission.get_param_indices('sigma2')],
                           np.trim_zeros(theta[:, M.emission.get_param_indices('sigma2')], 'b'), name='sigma2')
    print('Done.')


def _simulate_full_GME(K=5, P=1000, N=20, num_sub=10, max_iter=100):
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionT = MixGaussianExp(K=K, N=N, P=P)
    # emissionT.random_params()

    # Step 2: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=num_sub)
    Y, signal = emissionT.sample(U)

    # Step 3: Compute the log likelihood from the true model
    theta_true = np.concatenate([emissionT.get_params(), arrangeT.get_params()])
    emissionT.initialize(Y)
    emll_true = emissionT.Estep(signal=signal)
    Uhat, ll_a = arrangeT.Estep(emll_true)
    loglike_true = pt.sum(Uhat * emll_true) + pt.sum(ll_a)

    # Step 4: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionM = MixGaussianExp(K=K, N=N, P=P)
    # emissionM.set_params(pt.cat((emissionM.V.flatten(), emissionT.sigma2.reshape(1), emissionM.alpha.reshape(1), emissionM.beta.reshape(1))))

    # Step 5: Estimate the parameter thetas to fit the new model using EM
    M = FullModel(arrangeM, emissionM)
    M, ll, theta, _ = M.fit_em(Y=Y, iter=max_iter, tol=0.0001)

    # Plotfitting results
    iters = np.trim_zeros(ll, 'b').size
    _plot_loglike(np.trim_zeros(ll, 'b'), loglike_true, color='b')
    true_V = theta_true[emissionT.get_param_indices('V')].reshape(N, K)
    predicted_V = theta[0:iters, M.emission.get_param_indices('V')]
    idx = matching_params(true_V, predicted_V, once=False)

    _plot_diff(true_V, predicted_V, index=idx, name='V')
    _plt_single_param_diff(theta_true[M.emission.get_param_indices('sigma2')],
                           np.trim_zeros(theta[:, M.emission.get_param_indices('sigma2')], 'b'), name='sigma2')
    _plt_single_param_diff(theta_true[M.emission.get_param_indices('beta')],
                           np.trim_zeros(theta[:, M.emission.get_param_indices('beta')], 'b'), name='beta')
    # SSE = mean_adjusted_sse(Y, M.emission.V, U_hat, adjusted=True, soft_assign=False)
    print('Done.')


def _simulate_full_VMF(K=5, P=100, N=40, num_sub=10, max_iter=50):
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionT = MixVMF(K=K, N=N, P=P, uniform=True)
    # emissionT.random_params()

    # Step 2: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=num_sub)
    Y = emissionT.sample(U)

    # Step 3: Compute the log likelihood from the true model
    theta_true = np.concatenate([emissionT.get_params(), arrangeT.get_params()])
    T = FullModel(arrangeT, emissionT)
    Uhat, loglike_true = T.Estep(Y=Y) # Run only Estep! 

    # Step 4: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False, remove_redundancy=False)
    emissionM = MixVMF(K=K, N=N, P=P, uniform=True)
    # emissionM.set_params([emissionM.V, emissionM.kappa])

    # Step 5: Estimate the parameter thetas to fit the new model using EM
    M = FullModel(arrangeM, emissionM)
    M, ll, theta, _ = M.fit_em(Y=Y, iter=max_iter, tol=0.00001)

    # Plotfitting results
    iters = np.trim_zeros(ll, 'b').size
    _plot_loglike(np.trim_zeros(ll, 'b'), loglike_true, color='b')
    true_V = theta_true[emissionT.get_param_indices('V')].reshape(N, K)
    predicted_V = theta[0:iters, M.emission.get_param_indices('V')]
    idx = matching_params(true_V, predicted_V, once=False)

    _plot_diff(true_V, predicted_V, index=idx, name='V')
    _plot_diff(theta_true[M.emission.get_param_indices('kappa')].reshape(1, K),
               theta[0:iters, M.emission.get_param_indices('kappa')], name='kappa')
    print('Done.')


if __name__ == '__main__':
    # _simulate_full_VMF(K=5, P=1000, N=40, num_sub=10, max_iter=100)
    # _simulate_full_GMM(K=5, P=1000, N=20, num_sub=10, max_iter=100)
    _simulate_full_GME(K=5, P=2000, N=20, num_sub=10, max_iter=100)

