#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/14/2021
Emission models class

Author: dzhi, jdiedrichsen
"""
import numpy as np
import torch as pt
from scipy import stats, special
from torch import log, exp, sqrt
from generativeMRF.sample_vmf import random_VMF
from generativeMRF.model import Model
from generativeMRF.depreciated.AIS_test import rejection_sampling
import generativeMRF.arrangements as ar

class EmissionModel(Model):
    """ Abstract class for emission models
    """
    def __init__(self, K=4, N=10, P=20, data=None, X=None):
        """ Abstract constructor of emission models
        Args:
            K: the number of clusters
            N: the number of observations if given
            P: the number of brain locations
            data: the data, shape (num_subj, N, P)
            X: the design matrix of observations,
               shape of (N, M) tensor
        """
        self.K = K  # Number of states
        self.P = P
        self.nparams = 0
        self.PI = pt.tensor(pt.pi, dtype=pt.get_default_dtype())
        if X is not None:
            if type(X) is np.ndarray:
                X = pt.tensor(X, dtype=pt.get_default_dtype())
            self.X = X
            self.N = X.shape[0]
            self.M = X.shape[1]
        else:
            self.N = N
            self.M = N
            self.X = pt.eye(self.N)

        if data is not None:
            self.initialize(data)
        self.tmp_list=['Y']
    
    def initialize(self, data, X=None):
        """ Initializes the emission model with data set.
            The data are stored in the object itself
            call clear() to remove.

        Args:
            data (pt.tensor, ndarray): numsubj x N x P data tensor
            X (array, optional): Design matrix. Defaults to None.
        
        """
        if type(data) is np.ndarray:
            data = pt.tensor(data, dtype=pt.get_default_dtype())
        elif type(data) is pt.Tensor:
            pass
        else:
            raise ValueError("The input data must be a numpy.array or torch.tensor.")

        if X is not None:
            if type(X) is np.ndarray:
                X = pt.tensor(X, dtype=pt.get_default_dtype())
            assert X.shape == self.X.shape, "Input X mut have same shape of self.X"
            self.X = X
        assert self.X.shape[0] == data.shape[1], "data must has same number of observations in X"

        self.Y = data  # This is assumed to be (num_sub,P,N)
        self.num_subj = data.shape[0]

    def Estep(self, sub=None):
        """ Implemnents E-step and returns

        Args:
            sub (list):
                List of indices of subjects to use. Default=all (None)

        Returns:
            emloglik (np.array):
                emission log likelihood log p(Y|u,theta_E) a numsubjxPxK matrix
        """
        pass

    def Mstep(self, U_hat):
        """ Implements M-step for the model
        """
        pass

    def random_params(self):
        """ Sets parameters to random values
        """
        pass

class MultiNomial(EmissionModel):
    """ Multinomial emission model with coupling strength theta_s
    """
    def __init__(self, K=4, P=20, params=None):
        super().__init__(K, 1, P)
        self.w = pt.tensor(1.0)
        self.set_param_list(['w'])
        self.name = 'MN'
        self.V = pt.eye(K) # This is for consistency only , so the model can be evaluated on test data (with cos err) 
        if params is not None:
            self.set_params(params)

    def initialize(self, Y):
        """ Stores the data in emission model itself
            Calculates sufficient stats on the data that does not depend on u,
            and allocates memory for the sufficient stats that does.
        """
        self.Y = Y

    def Estep(self, Y=None, sub=None):
        """ Estep: Returns log p(Y|U) for each value of U, up to a constant
            Collects the sufficient statistics for the M-step specify which
            subject to optimize

        Args:
            Y (pt.tensor, optional): numsubj x N x P data tensor. Defaults to None.
            sub (list, optional): List of indices of subjects to use. Defaults to None.

        Returns:
            LL (pt.tensor): the expected log likelihood for emission model,
                shape (nSubject * K * P)
        """
        if Y is not None:
            self.initialize(Y)
        n_subj = self.Y.shape[0]

        LL = self.Y * self.w - log(self.K-1+exp(self.w))
        return LL

    def Mstep(self, U_hat):
        """ Performs the M-step on a specific U-hat. In this emission model,
            the parameters need to be updated are V and sigma2.

        Args:
            U_hat (pt.tensor): the posterior mean of U, shape (nSubject * K * P)
        """
        mean_uy = pt.mean(pt.sum(self.Y * self.U_hat, dim=1)) # this is E(yTu)
        self.w = log(1-self.K+(self.K-1)/(1-mean_uy)) 

    def sample(self, U, signal=None):
        """ Generate random data given this emission model
        Args:
            U (pt.tensor): The prior arrangement U from arrangement model

        Returns:
            Y (pt.tensor): sampled data Y (compressed form)
        """
        Ue = ar.expand_mn(U,self.K)
        p = pt.softmax(Ue*self.w,1)
        Y = ar.sample_multinomial(p, kdim=1, compress=False)
        return Y

class MixGaussian(EmissionModel):
    """ Mixture of Gaussians with isotropic noise
    """
    def __init__(self, K=4, N=10, P=20, data=None, X=None,
                 params=None, std_V=True):
        """ Constructor for the emission model

        Args:
            K (int, optional): Number of parcels.
            N (int, optional): Number of tasks.
            P (int, optional): Number of brain voxels.
            data (pt.tensor, optional): numsubj x N x P data tensor.
                Defaults to None.
            X (pt.tensor, optional): Design matrix of the tasks.
                Defaults to None.
            params (dict, optional): Dictionary of parameters.
                Defaults to None.
            std_V (bool, optional): Whether to standardize the Vs.
        """
        super().__init__(K, N, P, data, X)
        self.std_V = std_V
        self.random_params()
        self.set_param_list(['V', 'sigma2'])
        self.name = 'GMM'
        if params is not None:
            self.set_params(params)
        self.tmp_list=['Y','rss','YY']

    def initialize(self, data, X=None):
        """ Stores the data in emission model itself. Calculates
            sufficient stats on the data that does not depend on u,
            and allocates memory for the sufficient stats that does.

        Args:
            data (pt.tensor): numsubj x N x P data tensor
            X (pt.tensor, optional): Design matrix of the tasks.
        """
        super().initialize(data, X=X)
        self.YY = self.Y**2
        self.rss = pt.empty((self.num_subj, self.K, self.P))

    def random_params(self):
        """ In this mixture gaussians, the parameters are parcel-specific
            mean V_k and variance. Here, we assume the variance is equally
            across different parcels. Therefore, there are total k+1
            parameters in this mixture model. We set the initial random
            parameters for gaussian mixture here.
        """
        self.V = pt.randn(self.M, self.K)/np.sqrt(self.M)
        if self.std_V:  # standardise V to unit length
            # Not clear why this should be constraint for GMM, but ok
            self.V = self.V / pt.sqrt(pt.sum(self.V**2, dim=0))
        self.sigma2 = pt.tensor(np.exp(np.random.normal(0, 0.3)),
                                dtype=pt.get_default_dtype())

    def Estep(self, Y=None, sub=None):
        """ Estep: Returns log p(Y|U) for each value of U, up to a constant
            Collects the sufficient statistics for the M-step specify
            which subject to optimize

        Args:
            Y (pt.tensor, optional): numsubj x N x P data tensor.
                Defaults to None.
            sub (list, optional): List of indices of subjects to use.
                Defaults to None.

        Return:
            LL (pt.tensor): the expected log likelihood for emission model,
                shape (nSubject * K * P)
        """
        if Y is not None:
            self.initialize(Y)
        n_subj = self.Y.shape[0]
        if sub is None:
            sub = range(self.Y.shape[0])

        LL = pt.empty((self.Y.shape[0], self.K, self.P))
        # This is u.T V.T V u for each u
        uVVu = pt.sum(pt.matmul(self.X, self.V)**2, dim=0)
        YV = pt.matmul(pt.matmul(self.X, self.V).T, self.Y)
        self.rss = pt.sum(self.YY, dim=1, keepdim=True) \
                   - 2*YV + uVVu.reshape((self.K, 1))
        LL = - 0.5 * self.N*(log(self.PI) + log(self.sigma2)) \
             - 0.5 / self.sigma2 * self.rss

        return pt.nan_to_num(LL)

    def Mstep(self, U_hat):
        """ Performs the M-step on a specific U-hat. In this emission model,
            the parameters need to be updated are V and sigma2.
        """
        regressX = pt.matmul(pt.linalg.inv(pt.matmul(self.X.T, self.X)), self.X.T)  # (N, M)
        nan_voxIdx = self.Y[:, 0, :].isnan().unsqueeze(1).repeat(1, self.K, 1)
        YU = pt.matmul(pt.nan_to_num(self.Y), pt.transpose(U_hat, 1, 2))

        # 1. Here we update the v_k, which is sum_i(Uhat(k)*Y_i) / sum_i(Uhat(k))
        this_U_hat = pt.clone(U_hat)
        this_U_hat[nan_voxIdx] = 0
        self.V = pt.matmul(regressX, pt.sum(YU, dim=0)/this_U_hat.sum(dim=(0, 2)))
        if self.std_V:
            self.V = self.V / pt.sqrt(pt.sum(self.V ** 2, dim=0))

        # 2. Updating sigma2 (rss is calculated using updated V)
        YV = pt.matmul(pt.matmul(self.X, self.V).T, self.Y)
        # JD: Again, you should be able to avoid looping over subjects here entirely.
        ERSS = pt.sum(self.YY, dim=1, keepdim=True) - 2 * YV + \
               pt.sum(pt.matmul(self.X, self.V)**2, dim=0).view((self.K, 1))
        self.sigma2 = pt.nansum(this_U_hat * ERSS) / (self.N * self.P * self.num_subj)

    def sample(self, U, signal=None):
        """ Generate random data given this emission model

        Args:
            U (pt.tensor): prior arrangement U from arrangement model.
                numsubj x N x P tensor of assignments

        Returns:
            Y (pt.tensor): numsubj x N x P tensor of sample data
        """
        if type(U) is np.ndarray:
            U = pt.tensor(U, dtype=pt.int)
        elif type(U) is pt.Tensor:
            U = U.int()
        else:
            raise ValueError('The given U must be numpy ndarray or torch Tensor!')

        num_subj = U.shape[0]
        Y = pt.normal(0, pt.sqrt(self.sigma2), (num_subj, self.N, self.P))
        for s in range(num_subj):
            # And the V_k given by the U, then X*V_k*U = (n_sub, N, P)
            Y[s, :, :] = Y[s, :, :] + pt.matmul(self.X, self.V[:, U[s, :].long()])

        return Y


class MixGaussianExp(EmissionModel):
    """ Mixture of Gaussians with signal strength (fit gamma distribution)
        for each voxel. Scaling factor on the signal and a fixed noise variance
    """
    def __init__(self, K=4, N=10, P=20, num_signal_bins=88, data=None, X=None, params=None,
                 std_V=True, type_estep='linspace'):
        super().__init__(K, N, P, data, X)
        self.std_V = std_V  # Standardize mean vectors?
        self.random_params()
        self.set_param_list(['V', 'sigma2', 'alpha', 'beta'])
        self.name = 'GME'
        if params is not None:
            self.set_params(params)
        self.num_signal_bins = num_signal_bins  # Bins for approximation of signal strength
        self.type_estep = type_estep  # Added for a period until we have the best technique
        self.tmp_list=['Y','YY','s','s2','rss']

    def initialize(self, data, X=None):
        """ Stores the data in emission model itself. Calculates sufficient stats
            on the data that does not depend on u, and allocates memory for the
            sufficient stats that does.

        Args:
            data (pt.tensor): numsubj x N x P tensor of data
            X (pt.tensor): numsubj x N x M tensor of design matrix
        """
        super().initialize(data, X=X)
        self.YY = self.Y ** 2
        self.maxlength = pt.max(pt.sqrt(pt.sum(self.YY, dim=1)).nan_to_num())
        self.s = pt.empty((self.num_subj, self.K, self.P))
        self.s2 = pt.empty((self.num_subj, self.K, self.P))
        self.rss = pt.empty((self.num_subj, self.K, self.P))

    def random_params(self):
        """ In this mixture gaussians, the parameters are parcel-specific mean V_k
            and variance. Here, we assume the variance is equal across different parcels.
            Therefore, there are total k+1 parameters in this mixture model
            set the initial random parameters for gaussian mixture
        """
        self.V = pt.randn(self.M, self.K)
        if self.std_V:  # standardise V to unit length
            self.V = self.V / pt.sqrt(pt.sum(self.V ** 2, dim=0))
        self.sigma2 = pt.tensor(np.exp(np.random.normal(0, 0.3)), dtype=pt.get_default_dtype())
        self.alpha = pt.tensor(1, dtype=pt.get_default_dtype())
        self.beta = pt.tensor(1, dtype=pt.get_default_dtype())

    def Estep_max(self, Y=None, sub=None):
        """ Estep: Returns log p(Y, s|U) for each value of U, up to a constant
            Collects the sufficient statistics for the M-step

        Args:
            Y (pt.tensor): numsubj x N x P tensor of real data
            sub (list): specify which subject to optimize

        Returns:
            LL (pt.tensor): the expected log likelihood for emission model.
                shape of numsubj x K x P tensor
        """
        if Y is not None:
            self.initialize(Y)
        if sub is None:
            sub = range(self.Y.shape[0])

        LL = pt.empty((self.Y.shape[0], self.K, self.P))
        uVVu = pt.sum(self.V ** 2, dim=0)  # This is u.T V.T V u for each u

        for i in sub:
            YV = pt.mm(self.Y[i, :, :].T, self.V)
            self.s[i, :, :] = (YV / uVVu - self.beta * self.sigma2).T  # Maximized g
            self.s[i][self.s[i] < 0] = 0  # Limit to 0
            self.rss[i, :, :] = pt.sum(self.YY[i, :, :], dim=0) - 2 * YV.T * self.s[i, :, :]\
                                + self.s[i, :, :] ** 2 * uVVu.reshape((self.K, 1))
            # the log likelihood for emission model (GMM in this case)
            LL[i, :, :] = - 0.5 * self.N * (log(2*self.PI) + pt.log(self.sigma2)) \
                          - 0.5 * (1 / self.sigma2) * self.rss[i, :, :] \
                          + self.alpha * pt.log(self.beta) - \
                          pt.special.gammaln(self.alpha) - self.beta * self.s[i, :, :]

        return LL

    def Estep_old(self, Y=None, signal=None, sub=None):
        """ Approximate the log p(Y, s|U) for each value of U by sampling,
            up to a constant. Collects the sufficient statistics for the M-step

        Args:
            sub: specify which subject to optimize

        Returns:
             LL (pt.tensor): the expected log likelihood for emission model,
                shape (nSubject * K * P)
        """
        if Y is not None:
            self.initialize(Y)

        if sub is None:
            sub = range(self.Y.shape[0])

        LL = pt.empty((self.Y.shape[0], self.K, self.P))
        uVVu = pt.sum(self.V ** 2, dim=0)  # This is u.T V.T V u for each u

        for i in sub:
            YV = pt.mm(self.Y[i, :, :].T, self.V)
            # Importance sampling from p(s_i|y_i, u_i).
            # First try sample from uniformed distribution
            # plt.figure()
            if signal is None:
                for k in range(self.K):
                    for p in range(self.P):
                        # Here try to sampling the posterior of p(s_i|y_i, u_i)
                        # for each given y_i and u_i(k)
                        x = pt.linspace(0,self.maxlength*1.1,77)
                        loglike = - 0.5 * (1 / self.sigma2) \
                                  * (-2 * YV[p, k] * x + uVVu[k] * x ** 2) - self.beta * x
                        # This is the posterior prob distribution of p(s_i|y_i,u_i(k))
                        post = loglik2prob(loglike)
                        self.s[i, k, p] = pt.sum(x * post)
                        self.s2[i, k, p] = pt.sum(x**2 * post)
                        if pt.isnan(self.s[i, k, p]) or pt.isnan(self.s2[i, k, p]):
                            print(i, k, p)
                        # plt.plot(x, post)
                    # plt.show()
            else:
                self.s = signal.unsqueeze(1).repeat(1, self.K, 1)
                self.s2 = signal.unsqueeze(1).repeat(1, self.K, 1)**2

            self.rss[i, :, :] = pt.sum(self.YY[i, :, :], dim=0) - 2 * YV.T * self.s[i, :, :] \
                                + self.s2[i, :, :] * uVVu.reshape((self.K, 1))
            # the log likelihood for emission model (GMM in this case)
            LL[i, :, :] = - 0.5 * self.N * (log(2*self.PI) + pt.log(self.sigma2)) \
                          - 0.5 * (1 / self.sigma2) * self.rss[i, :, :] \
                          + pt.log(self.beta) - self.beta * self.s[i, :, :]

        return LL

    def Estep_ais(self, Y=None, signal=None, sub=None):
        """ Approximate the log p(Y, s|U) for each value of U using AIS
            (annealed importance sampling) up to a constant.
            Collects the sufficient statistics for the M-step

        Args:
            Y (pt.tensor): shape (nSubject * N * P)
            signal (pt.tensor): shape (nSubject * nTime)
            sub: specify which subject to optimize

        Returns:
            LL (pt.tensor): the expected log likelihood for emission model,
                shape (nSubject * K * P)
        """
        if Y is not None:
            self.initialize(Y)

        if sub is None:
            sub = range(self.Y.shape[0])

        LL = pt.empty((self.Y.shape[0], self.K, self.P))
        uVVu = pt.sum(self.V ** 2, dim=0)  # This is u.T V.T V u for each u

        for i in sub:
            YV = pt.mm(self.Y[i, :, :].T, self.V)
            # Importance sampling from p(s_i|y_i, u_i).
            # First try sample from uniformed distribution
            # plt.figure()
            if signal is None:
                f_n = lambda x: pt.exp(-(x - 1) ** 2 / (2 * 2 ** 2))
                q_n = pt.distributions.normal.Normal(3, 3)
                for k in range(self.K):
                    for p in range(self.P):
                        # Here try to sampling the posterior of p(s_i|y_i, u_i)
                        # for each given y_i and u_i(k)
                        x = pt.linspace(0,self.maxlength*1.1,77)
                        loglike = lambda a: -0.5*(1/self.sigma2) \
                                            * (-2*YV[p, k]*a + uVVu[k]*a**2) - self.beta*a
                        likelihood = lambda a: exp(-0.5*(1/self.sigma2)
                                                   * (-2*YV[p, k]*a + uVVu[k]*a**2) - self.beta*a)

                        res1, res2 = annealed_importance_sampling(likelihood, f_n, q_n,
                                                                  num_sample=100, interval=10)

                        # This is the posterior prob distribution of p(s_i|y_i,u_i(k))
                        # post = loglik2prob(loglike)
                        self.s[i, k, p] = res1
                        self.s2[i, k, p] = res2
                        if pt.isnan(self.s[i, k, p]) or pt.isnan(self.s2[i, k, p]):
                            print(i, k, p)
                        # plt.plot(x, post)
                    # plt.show()

            else:
                self.s = signal.unsqueeze(1).repeat(1, self.K, 1)
                self.s2 = signal.unsqueeze(1).repeat(1, self.K, 1)**2

            self.rss[i, :, :] = pt.sum(self.YY[i, :, :], dim=0) - 2 * YV.T * self.s[i, :, :] \
                                + self.s2[i, :, :] * uVVu.reshape((self.K, 1))
            # the log likelihood for emission model (GMM in this case)
            LL[i, :, :] = - 0.5 * self.N * (pt.log(2*self.PI) + pt.log(self.sigma2)) \
                          - 0.5 * (1 / self.sigma2) * self.rss[i, :, :] \
                          + pt.log(self.beta) - self.beta * self.s[i, :, :]

        return pt.nan_to_num(LL)

    def Estep_import(self, Y=None, signal=None):
        """ Estep using importance sampling from exp(beta): Sampling is done per
            iteration for all voxels and clusters. The weighting is simply
            the p(y|s,u), as the p(s) is already done in the sampling

        Args:
            Y (pt.tensor): shape (nSubject * N * P)
            sub: specify which subject to optimize

        Returns:
            LL (pt.tensor): the expected log likelihood for emission model,
                shape (nSubject * K * P)
        """
        if Y is not None:
            self.initialize(Y)

        n_subj = self.Y.shape[0]
        LL = pt.empty((n_subj, self.K, self.P))
        uVVu = pt.sum(pt.matmul(self.X, self.V)**2, dim=0)  # This is u.T V.T V u for each u
        YV = pt.matmul(pt.matmul(self.X, self.V).T, self.Y)

        # Instead of evenly spaced, sample from exp(beta)
        dist = pt.distributions.exponential.Exponential(self.beta)
        x = dist.sample((self.num_signal_bins,))

        logpi = pt.log(pt.tensor(2*pt.pi, dtype=pt.get_default_dtype()))
        if signal is None:
            # This is p(y,s|u)
            loglike = - 0.5/self.sigma2 * (-2 * YV.view(n_subj,self.K,self.P,1) * x
                                           + uVVu.view(self.K,1,1) * (x ** 2))
            # This is the posterior prob distribution of p(s_i|y_i,u_i(k))
            post = pt.softmax(loglike, dim=3)
            self.s = pt.sum(x * post, dim=3)
            self.s2 = pt.sum(x**2 * post, dim=3)
        else:
            self.s = signal.unsqueeze(1).repeat(1, self.K, 1)
            self.s2 = signal.unsqueeze(1).repeat(1, self.K, 1)**2

        self.rss = pt.sum(self.YY, dim=1, keepdim=True) - 2 * YV * self.s \
                   + self.s2 * uVVu.reshape((self.K, 1))
        # the log likelihood for emission model (GMM in this case)
        LL = -0.5 * self.N * (logpi + pt.log(self.sigma2)) - 0.5 / self.sigma2 * self.rss \
             + pt.log(self.beta) - self.beta * self.s

        return pt.nan_to_num(LL)

    def Estep_reject(self, Y=None, signal=None):
        """ Estep using reject sampling from exp(beta): Sampling is done per
            iteration for all voxels and clusters. The weighting is simply
            the p(y|s,u), as the p(s) is already done in the sampling

        Args:
            Y (tensor): the data, shape (nSubject, N, P)
            sub: specify which subject to optimize

        Returns:
            LL (pt.tensor): the expected log likelihood for emission model,
                shape (nSubject * K * P)
        """
        if Y is not None:
            self.initialize(Y)

        n_subj = self.Y.shape[0]
        uVVu = pt.sum(pt.matmul(self.X, self.V)**2, dim=0)  # This is u.T V.T V u for each u
        YV = pt.matmul(pt.matmul(self.X, self.V).T, self.Y)

        # Instead of evenly spaced, sample from q(x) (here is exp(beta))
        dist = pt.distributions.exponential.Exponential(self.beta)
        x = dist.sample((self.num_signal_bins,))
        qx_pdf = dist.log_prob(x).exp()

        if signal is None:
            # This is p(y,s|u)
            loglike = - 0.5/self.sigma2 * (-2 * YV.view(n_subj,self.K,self.P,1) * x
                                           + uVVu.view(self.K,1,1) * (x ** 2)) - self.beta * x
            likelihood = pt.softmax(loglike, dim=3)

            # Here to compute the ratio of p(x)/q(x) as the weights for each sample
            # The p(x) is the true distribution which can be either normalized or un-normalized
            M, _ = pt.max(likelihood / qx_pdf, dim=3, keepdim=True)
            ratio = likelihood / (M*qx_pdf)

            u = pt.distributions.uniform.Uniform(0, 1).sample(ratio.shape)
            mask = pt.where(u <= ratio, 1.0, 0.0)
            # This is the posterior prob distribution of p(s_i|y_i,u_i(k))
            # post = pt.softmax(ratio, dim=3)
            self.s = pt.sum(x * mask, dim=3)/mask.sum(dim=3)
            self.s2 = pt.sum(x**2 * mask, dim=3)/mask.sum(dim=3)
        else:
            self.s = signal.unsqueeze(1).repeat(1, self.K, 1)
            self.s2 = signal.unsqueeze(1).repeat(1, self.K, 1)**2

        self.rss = pt.sum(self.YY, dim=1, keepdim=True) - 2 * YV * self.s \
                   + self.s2 * uVVu.reshape((self.K, 1))
        # the log likelihood for emission model (GMM in this case)
        LL = -0.5 * self.N * (pt.log(2*self.PI) + pt.log(self.sigma2)) - 0.5/self.sigma2*self.rss \
             + pt.log(self.beta) - self.beta * self.s

        return pt.nan_to_num(LL)

    def Estep_linspace(self, Y=None, signal=None):
        """ Estep sampling using simple linear sapcing from exp(beta): Sampling
            is done per iteration for all voxels and clusters. The weighting is simply
            the p(y|s,u), as the p(s) is already done in the sampling

        Args:
            Y (tensor): the data, shape (nSubject, N, P)
            sub: specify which subject to optimize

        Returns:
            LL (pt.tensor): the expected log likelihood for emission model,
                shape (nSubject * K * P)
        """
        if Y is not None:
            self.initialize(Y)

        n_subj = self.Y.shape[0]
        LL = pt.empty((n_subj, self.K, self.P))
        uVVu = pt.sum(pt.matmul(self.X, self.V)**2, dim=0)  # This is u.T V.T V u for each u

        YV = pt.matmul(pt.matmul(self.X, self.V).T, self.Y)
        signal_max = self.maxlength*1.2  # Make the maximum signal 1.2 times the max data magnitude
        signal_bin = signal_max / self.num_signal_bins
        x = pt.linspace(signal_bin/2, signal_max, self.num_signal_bins)
        logpi = pt.log(pt.tensor(2*pt.pi, dtype=pt.get_default_dtype()))
        if signal is None:
            # This is p(y,s|u)
            loglike = - 0.5/self.sigma2 * (-2*YV.view(n_subj,self.K,self.P,1)*x
                                           + uVVu.view(self.K,1,1) * (x**2)) - self.beta*x
            # This is the posterior prob distribution of p(s_i|y_i,u_i(k))
            post = pt.softmax(loglike, dim=3)
            self.s = pt.sum(x * post, dim=3)
            self.s2 = pt.sum(x**2 * post, dim=3)
        else:
            self.s = signal.unsqueeze(1).repeat(1, self.K, 1)
            self.s2 = signal.unsqueeze(1).repeat(1, self.K, 1)**2

        self.rss = pt.sum(self.YY, dim=1, keepdim=True) - 2 * YV * self.s \
                   + self.s2 * uVVu.reshape((self.K, 1))
        # the log likelihood for emission model (GMM in this case)
        LL = -0.5 * self.N * (logpi + pt.log(self.sigma2)) - 0.5 / self.sigma2 * self.rss \
             + pt.log(self.beta) - self.beta * self.s

        return pt.nan_to_num(LL)

    def Estep_mcmc_hasting(self, Y=None, signal=None, iters=10):
        """ Estep using MCMC: Sampling is done per iteration for all voxels and clusters
            The weighting is simply the p(y|s,u), as the p(s) is already done in the sampling

        Args:
            Y (tensor): the data, shape (nSubject, N, P)
            signal (tensor): the signal, shape (nSubject, K, P)
            iters (int): number of iterations for MCMC

        Returns:
            LL (pt.tensor): the expected log likelihood for emission model,
                shape (nSubject * K * P)
        """
        if Y is not None:
            self.initialize(Y)

        n_subj = self.Y.shape[0]
        LL = pt.empty((n_subj, self.K, self.P))
        uVVu = pt.sum(pt.matmul(self.X, self.V)**2, dim=0)  # This is u.T V.T V u for each u
        YV = pt.matmul(pt.matmul(self.X, self.V).T, self.Y)

        # The unnormalized distribution that we want to compute the expecation
        p_x = lambda a: exp(- 0.5 / self.sigma2 * (-2 * YV.view(n_subj, self.K, self.P, 1) * a +
                        uVVu.view(self.K, 1, 1) * (a ** 2)) - self.beta * a)
        x_t = pt.full((n_subj, self.K, self.P, 1), self.beta)

        if signal is None:
            for i in range(iters):
                # Sample x' ~ q(x'|x_t)
                x_prime = 1 / pt.distributions.exponential.Exponential(x_t).sample()
                ll_proposal = pt.distributions.exponential.Exponential(x_prime).log_prob(x_t).exp()
                ll_current = pt.distributions.exponential.Exponential(x_t).log_prob(x_prime).exp()
                acc_rate = (ll_proposal*p_x(x_prime)) / (ll_current*p_x(x_t))

                random_u = pt.distributions.uniform.Uniform(0, 1).sample(acc_rate.size())
                x_t = pt.where(random_u <= acc_rate, x_prime, x_t)
                self.s = x_t.squeeze(3)
                self.s2 = x_t.squeeze(3)**2
        else:
            self.s = signal.unsqueeze(1).repeat(1, self.K, 1)
            self.s2 = signal.unsqueeze(1).repeat(1, self.K, 1) ** 2

        self.rss = pt.sum(self.YY, dim=1, keepdim=True) - 2 * YV * self.s \
                   + self.s2 * uVVu.reshape((self.K, 1))
        # the log likelihood for emission model (GMM in this case)
        LL = -0.5 * self.N * (log(2*self.PI) + pt.log(self.sigma2)) - 0.5 / self.sigma2 * self.rss \
             + pt.log(self.beta) - self.beta * self.s

        return pt.nan_to_num(LL)

    def Estep(self, Y=None, signal=None):
        """ Performs the E-step on the data Y and signal s using different methods

        Args:
            Y (tensor): the data, shape (nSubject, N, P)
            signal (tensor): the signal, shape (nSubject, K, P)

        Returns:
            LL (pt.tensor): the expected log likelihood for emission model,
        """
        if self.type_estep == 'linspace':
            return self.Estep_linspace(Y, signal)
        elif self.type_estep == 'import':
            return self.Estep_import(Y, signal)
        elif self.type_estep == 'reject':
            return self.Estep_reject(Y, signal)
        elif self.type_estep == 'ais':
            return self.Estep_ais(Y, signal)
        elif self.type_estep == 'mcmc':
            return self.Estep_mcmc_hasting(Y, signal)
        else:
            raise NameError('An E-step method must be given.')

    def Mstep(self, U_hat):
        """ Performs the M-step on a specific U-hat. U_hat = E[u_i ^(k), s_i]
            In this emission model, the parameters need to be updated
            are V, sigma2, alpha, and beta

        Args:
            U_hat: The expected emission log likelihood
        """
        regressX = pt.matmul(pt.linalg.inv(pt.matmul(self.X.T, self.X)), self.X.T)  # (N, M)
        nan_voxIdx = self.Y[:, 0, :].isnan().unsqueeze(1).repeat(1, self.K, 1)
        this_U_hat = pt.clone(U_hat)
        this_U_hat[nan_voxIdx] = 0
        YU = pt.matmul(pt.nan_to_num(self.Y), pt.transpose(U_hat * self.s, 1, 2).nan_to_num())
        US = (this_U_hat * self.s).nansum(dim=2)
        US2 = (this_U_hat * self.s2).nansum(dim=2)


        # 1. Updating the V - standardize if requested
        # Here we update the v_k, which is sum_i(<Uhat(k), s_i>,*Y_i) / sum_i(Uhat(k), s_i^2)
        self.V = pt.matmul(regressX, pt.sum(YU, dim=0) / pt.sum(US2, dim=0))
        if self.std_V:
             self.V = self.V / pt.sqrt(pt.sum(self.V ** 2, dim=0))

        # 2. Updating the sigma squared.
        YV = pt.matmul(pt.matmul(self.X, self.V).T, self.Y)
        ERSS = pt.sum(self.YY, dim=1, keepdim=True) - 2 * YV * self.s + \
               self.s2 * pt.sum(pt.matmul(self.X, self.V)**2, dim=0).view((self.K, 1))
        self.sigma2 = pt.nansum(this_U_hat * ERSS) / (self.N * self.P * self.num_subj)

        # 3. Updating the beta (Since this is an exponential model)
        self.beta = self.P * self.num_subj / pt.sum(US)

    def sample(self, U, signal=None, return_signal=False):
        """ Generate random data given this emission model and parameters

        Args:
            U: The prior arrangement U from the arrangement model
            V: Given the initial V. If None, then randomly generate
            return_signal (bool): Return signal as well? False by default for compatibility

        Returns:
            Y (tensor): The generated sample data, shape (num_subj, N, P)
            signal (tensor, optional): The generated signal, shape (num_subj, P)
        """
        num_subj = U.shape[0]
        if signal is not None:
            np.testing.assert_equal((signal.shape[0], signal.shape[1]), (num_subj, self.P),
                                    err_msg='The given signal must with a shape of (num_subj, P)')
        else:
            # 1. Draw the signal strength for each node from an exponential distribution
            signal = pt.distributions.exponential.Exponential(self.beta).sample((num_subj, self.P))

        Y = pt.normal(0, pt.sqrt(self.sigma2), (num_subj, self.N, self.P))
        for s in range(num_subj):
            Y[s, :, :] = Y[s, :, :] + pt.matmul(self.X, self.V[:, U[s, :].long()]) * signal[s, :]

        # Only return signal when asked: compatibility with other models
        if return_signal:
            return Y, signal
        else:
            return Y


class MixVMF(EmissionModel):
    """ Mixture of Gaussians with isotropic noise
    """
    def __init__(self, K=4, N=10, P=20, data=None, 
                X=None, part_vec=None, params=None,
                uniform_kappa=True):
        """ Constructor for the vmf mixture emission model
        Args:
            K (int): the number of clusters
            N (int): the number of observations 
            P (int): the number of voxels
            data (ndarray,pt.tensor): n_sub x N x P  training data
            X (ndarray or tensor): N x M design matrix for task conditions 
            part_vec (ndarray or tensor): N-Vector indicating the number of the 
                      data partition (repetition).
                      Expample = [1,2,3,1,2,3,...] None: no data partition
            params: if None, no parameters to pass in.
                Otherwise take the passing parameters as the model params
            uniform_kappa: if True, the model learns a common kappa. Otherwise,
                           cluster-specific kappa
        """
        self.uniform_kappa = uniform_kappa
        if part_vec is not None:
            if isinstance(part_vec,(np.ndarray,pt.Tensor)):
                self.part_vec = pt.tensor(part_vec, dtype=pt.int)
            else:
                raise ValueError('Part_vec must be numpy ndarray or torch Tensor')
        else:
            self.part_vec = None
        
        super().__init__(K, N, P, data, X)
        self.random_params()
        self.set_param_list(['V', 'kappa'])
        self.name = 'VMF'
        if params is not None:
            self.set_params(params)
        self.tmp_list=['Y','num_part']

    def initialize(self, data):
        """ Calculates the sufficient stats on the data that does not depend on U,
            and allocates memory for the sufficient stats that does. For the VMF,
            it length-standardizes the data to length one. If part_vec is exist, then
            the raw data needs to be partitioned and normalize in each partition.
            After that, we restore Y to its original shape (num_sub, N, P). The new
            data for further fitting is X^T (shape M, N) * Y which has a shape
            (num_sub, M, P)
            Note: The shape of X (N, M) - N is # of observations, M is # of conditions

        Args:
            data: the input data array (or torch tensor). shape (n_subj, N, P)

        Returns: None. Store the data in emission model itself.

        Class attributes:
            self.num_part:  Number of available partitions per voxels. numsubj x 1 x P tensor 
                used in M step 
        """
        super().initialize(data)

        if self.part_vec is not None:
            # If self.part_vec is not None, meaning we need to split the data and making
            # normlization for partition specific data.
            assert (self.X.shape[0] == self.Y.shape[1]), \
                "When data partitioning happens, the design matrix X should have" \
                " same number of observations with input data Y."

            # Split the design matrix X and data and calculate (X^T*X)-1*X^T in each partition
            parts = pt.unique(self.part_vec)

            # Create array of new normalized data
            Y = pt.empty((len(parts),self.num_subj,self.M,self.P))
            for i,p in enumerate(parts):
                x = self.X[self.part_vec==p,:]
                # Y = (X^T@X)-1 @ X^T @ data: 
                # Use pinv (pseudo inverse here)- equivalent to :
                # pt.matmul(pt.linalg.inv(x.T @ x), x.T @ self.Y[:,self.part_vec==p,:])
                # But numerically more stable (i.e. when (xT @ x) is not invertible)
                Y[i,:,:,:] = pt.matmul(pt.linalg.pinv(x), self.Y[:,self.part_vec==p,:])

            # Length of vectors per partition, subject and voxel
            W = pt.sqrt(pt.sum(Y ** 2, dim=2, keepdim=True))
            # Keep track of how many available partions per voxels
            self.num_part = pt.sum(~W.isnan(),dim=0)

            # normalize in each partition
            Y = Y / W
            # Then sum over all the partitions
            self.Y = Y.nansum(dim=0)
            # Reshape back to (num_sub, M, P) - basically take the nansum across partitions
            self.M = self.Y.shape[1]
        else:
            # No data splitting
            # calculate (X^T*X)X^T*y to make the shape of Y is (num_sub, M, P)
            Y = pt.matmul(pt.linalg.pinv(self.X), self.Y)

            # calculate the data magnitude and get info of nan voxels
            W = pt.sqrt(pt.sum(Y ** 2, dim=1, keepdim=True)).unsqueeze(0)
            self.num_part = pt.sum(~W.isnan(), dim=0)

            # Normalized data with nan value
            self.Y = Y / pt.sqrt(pt.sum(Y ** 2, dim=1, keepdim=True))
            self.M = self.Y.shape[1]

    def random_params(self):
        """ In this mixture vmf model, the parameters are parcel-specific direction V_k
            and concentration value kappa_k.

        Returns: None, just passes the random parameters to the model
        """
        # standardise V to unit length
        V = pt.randn(self.M, self.K)
        self.V = V / pt.sqrt(pt.sum(V ** 2, dim=0))

        # VMF doesn't work properly for small kappa (let's say smaller than 8),
        # This is because the data will be very spread on the p-1 sphere, making the
        # model recovery difficult. Also, a small kappa cannot reflect to the real data
        # as the real parcels are likely to have concentrated within-parcel data.

        if self.uniform_kappa:
            self.kappa = pt.distributions.uniform.Uniform(10, 150).sample()
        else:
            self.kappa = pt.distributions.uniform.Uniform(10, 150).sample((self.K, ))

    def Estep(self, Y=None, sub=None):
        """ Estep: Returns log p(Y|U) for each voxel and value of U, 
            up to a constant. Collects the sufficient statistics for the M-step
        
        Args:
            Y (pt.tensor): Data (optional)
            sub (pt.tensor): vector of integer indices specify which subject to estimate (optional)

        Returns:
            LL (pt.tensor): the expected log likelihood for emission model,
            shape (nSubject * K * P)
        """
        if Y is not None:
            self.initialize(Y)

        if sub is None:
            sub = range(self.Y.shape[0])
        LL = pt.empty((self.Y.shape[0], self.K, self.P))

        # Calculate log-likelihood
        YV = pt.matmul(self.V.T, self.Y)
        PI = pt.tensor(pt.pi, dtype=pt.get_default_dtype())
        logCnK = (self.M/2 - 1)*log(self.kappa) - (self.M/2)*log(2*PI) - \
                 log_bessel_function(self.M/2 - 1, self.kappa)

        if self.uniform_kappa:
            LL = logCnK * self.num_part + self.kappa * YV
        else:
            LL = logCnK.unsqueeze(0).unsqueeze(2) * self.num_part + self.kappa.unsqueeze(1) * YV

        return pt.nan_to_num(LL[sub])

    def Mstep(self, U_hat):
        """ Performs the M-step on a specific U-hat. In this emission model,
            the parameters need to be updated are Vs (unit norm projected on
            the N-1 sphere) and kappa (concentration value).

        Args:
            U_hat: the expected log likelihood from the arrangement model

        Returns:
            Update all the object's parameters
        """
        if type(U_hat) is np.ndarray:
            U_hat = pt.tensor(U_hat, dtype=pt.get_default_dtype())

        # Multiply the expectation by the number of observations 
        JU_hat = self.num_part * U_hat

        # Calculate YU = \sum_i\sum_k<u_i^k>y_i and UU = \sum_i\sum_k<u_i^k>
        YU = pt.sum(pt.matmul(pt.nan_to_num(self.Y), pt.transpose(U_hat, 1, 2)), dim=0)
        UU = pt.sum(JU_hat, dim=0)

        # 1. Updating the V_k, which is || sum_i(Uhat(k)*Y_i) / sum_i(Uhat(k)) ||
        r_norm = pt.sqrt(pt.sum(YU ** 2, dim=0))
        self.V = YU / r_norm

        # 2. Updating kappa, kappa_k = (r_bar*N - r_bar^3)/(1-r_bar^2),
        # where r_bar = ||V_k||/N*Uhat
        if self.uniform_kappa:
            r_bar = r_norm.sum() / self.num_part.sum()
            # r_bar[r_bar > 0.95] = 0.95
            # r_bar[r_bar < 0.05] = 0.05
            r_bar = pt.mean(r_bar)
        else:
            r_bar = r_norm / pt.sum(UU, dim=1)
            # r_bar[r_bar > 0.95] = 0.95
            # r_bar[r_bar < 0.05] = 0.05

        self.kappa = (r_bar * self.M - r_bar**3) / (1 - r_bar**2)

    def sample(self, U, signal=None):
        """ Draw data sample from this model and given parameters

        Args:
            U(pt.tensor): num_subj x P arrangement for each subject
            signal(pt.tensor): num_subj x P signal for each subject

        Returns: The samples data from this distribution
        """
        if type(U) is np.ndarray:
            U = pt.tensor(U, dtype=pt.int)
        elif type(U) is pt.Tensor:
            U = U.int()
        else:
            raise ValueError('The given U must be numpy ndarray or torch Tensor!')

        if self.part_vec is None:
            num_parts = 1
            ind = [pt.arange(self.N)]
        else:
            parts = pt.unique(self.part_vec)
            num_parts = len(parts)
            ind = [self.part_vec == parts[j] for j in range(num_parts)]
    
        num_subj = U.shape[0]
        Y = pt.zeros((num_subj, self.N, self.P))

        for s in range(num_subj):
            par, counts = pt.unique(U[s], return_counts=True)
            for i, this_par in enumerate(par):
                y_full = []
                voxel_ind = pt.nonzero(U[s] == this_par).view(-1)
                # samples y shape (num_parts, P, M)
                y = pt.tensor(random_VMF(self.V[:, this_par].cpu().numpy(),
                                         self.kappa.cpu().numpy(),
                                         int(counts[i] * num_parts)),
                              dtype=pt.get_default_dtype())
                y = y.view(num_parts, counts[i], -1)
                # multiply within each partition
                y_full = pt.vstack([pt.matmul(self.X[ind[j], :], y[j].t())
                                    for j in range(num_parts)])
                Y[s, :, voxel_ind] = y_full

        return Y


class wMixVMF(MixVMF):
    """ Mixture of von-Mises Fisher distribution weighted by SNR. 
        This implementation follows the Model1: if a data point as a weight of w,
        we treat it as if it had been observed w times.
    """
    def __init__(self, K=4, N=10, P=20, data=None, X=None, part_vec=None,
                 params=None, uniform_kappa=True, weighting='lsquare'):
        """ Constructor for the wMixVMF class
        Args:
            K (int): the number of clusters
            N (int): the number of observations
            P (int): the number of voxels
            data (ndarray,pt.tensor): n_sub x N x P  training data
            X (ndarray or tensor): N x M design matrix for task conditions
            part_vec (ndarray or tensor): N-Vector indicating the number of the
                      data partition (repetition).
                      Expample = [1,2,3,1,2,3,...]None: no data partition
            params: if None, no parameters to pass in.
                Otherwise take the passing parameters as the model params
            uniform_kappa: if True, the model learns a common kappa. Otherwise,
                           cluster-specific kappa
            weighting: the weighting strategy, default 1 is data magnitude
        """
        super().__init__(K, N, P, data, X, part_vec, params, uniform_kappa)

        self.name = 'wVMF'
        self.weighting=weighting
        self.tmp_list=['Y','num_part','W']


    def initialize(self, data, weight=None):
        """ Calculates the sufficient stats on the data that does not depend on U,
            and allocates memory for the sufficient stats that does. For the VMF,
            it length-standardizes the data to length one. If part_vec is exist, then
            the raw data needs to be partitioned and normalize in each partition.
            After that, we restore Y to its original shape (num_sub, N, P). The new
            data for further fitting is X^T (shape M, N) * Y which has a shape
            (num_sub, M, P)
            Note: The shape of X (N, M) - N is # of observations, M is # of conditions

        Args:
            data: the input data array (or torch tensor). shape (n_subj, N, P)
            signal: pass in the weight for each data points is given

        Returns: 
            None. Store the data in emission model itself.

        Class attributes:
            self.num_part: the number of partitions (usually runs)
                           shape (n_subj, 1, P)
            self.W: the weights associated to each data points
        """
        super(MixVMF, self).initialize(data)

        if self.part_vec is not None:
            # If self.part_vec is not None, meaning we need to split the data and making
            # normlization for partition specific data.
            assert (self.X.shape[0] == self.Y.shape[1]), \
                "When data partitioning happens, the design matrix X should have" \
                " same number of observations with input data Y."

            # Split the design matrix X and data and calculate (X^T*X)-1*X^T in each partition
            parts = pt.unique(self.part_vec)

            # Create array of new normalized data
            Y = pt.empty((len(parts), self.num_subj, self.M, self.P))
            for i, p in enumerate(parts):
                x = self.X[self.part_vec == p, :]
                # Y = (X^T@X)-1 @ X^T @ data:
                # Use pinv (pseudo inverse here)- equivalent to :
                # pt.matmul(pt.linalg.inv(x.T @ x), x.T @ self.Y[:,self.part_vec==p,:])
                # But numerically more stable (i.e. when (xT @ x) is not invertible)
                Y[i, :, :, :] = pt.matmul(pt.linalg.pinv(x), self.Y[:, self.part_vec == p, :])

        else:
            # No data splitting
            # calculate (X^T*X)X^T*y to make the shape of Y is (num_sub, M, P)
            Y = pt.matmul(pt.linalg.pinv(self.X), self.Y).unsqueeze(0)

        # To consistent, the Y now has shape (part, n_subj, M, P) whether split or not
        # calculate the data magnitude
        l = pt.sqrt(pt.sum(Y ** 2, dim=2, keepdim=True))

        # Calculate the weighting per voxel, partition, and subject
        if weight is None:
            self.W = self._cal_weights(Y, type=self.weighting)
        else:
            self.W = weight.unsqueeze(0).unsqueeze(2)
        # Keep track of weighted length per voxel, fill 0 is missing voxel
        self.num_part = pt.sum(self.W, dim=0)

        # normalize in each partition, then multiply weights
        # Now, the self.W has shape (part, subj, 1, P)
        Y = (Y / l) * self.W
        # Then nansum over all the partitions
        self.Y = Y.nansum(dim=0)
        # Reshape back to (num_sub, M, P) - basically take the nansum across partitions
        self.M = self.Y.shape[1]

    def _cal_weights(self, Y, type='lsquare_sum2P'):
        """Calculate the weights per each of the independent observation
           (per voxel, partition, and subject)

        Args:
            Y (pt.Tensor): the input data, shape (partition, n_sub, M, P)
            type: the tupe to calculate the weights
                'length': the data raw magnitude
                'length_sum2P': the data raw magnitude and make them sum
                                to P in each partition
                'lsquare': the data raw magnitude squared
                'lsquare_sum2P': the data raw magnitude squared and make
                                 it sum to P ineach partition
                'length_normalized': the normalized magnitude squared to
                                     [0, 1]
                '+density': the normalized density [0, 1] + normalized
                            magnitude [0, 1]. So in total the range is
                            [0, 2]
                'ones': the weights are all 1, same as VMF

        Returns:
            the weights associated to each independent observation
        """
        d_magnitude = pt.sqrt(pt.sum(Y ** 2, dim=2, keepdim=True))
        num_part = pt.sum(~d_magnitude.isnan(), dim=0)
        d_magnitude = pt.nan_to_num(d_magnitude)
        if type == 'length':
            # weights are the magnitude
            return d_magnitude
        elif type == 'length_sum2P':
            # weights are the magnitude and make it
            # sum to P in each partition
            W = d_magnitude
            ratio = W.size(dim=3) / pt.sum(W, dim=3, keepdim=True)
            return W * ratio
        elif type == 'lsquare':
            # weights are the magnitude squared
            return d_magnitude ** 2
        elif type == 'lsquare_sum2P':
            # weights are the magnitude squared and make it
            # sum to P in each partition
            W = d_magnitude ** 2
            ratio = W.size(dim=3) / pt.sum(W, dim=3, keepdim=True)
            return W * ratio
        elif type == 'lsquare_sum2PJ':
            # weights are the magnitude squared and make it
            # sum to P*J globally
            W = d_magnitude ** 2
            ratio = W.numel() / W.sum()
            return W * ratio
        elif type == 'length_normalized':
            # weights are the normalized magnitude squared
            W = (d_magnitude - pt.min(d_magnitude, dim=3, keepdims=True)) / \
                (pt.max(d_magnitude, dim=3, keepdims=True) -
                    pt.min(d_magnitude, dim=3, keepdims=True))
            return W
        elif type == '+density':
            # weights are the density + magnitude
            W = (d_magnitude - pt.min(d_magnitude, dim=3, keepdims=True)) / \
                (pt.max(d_magnitude, dim=3, keepdims=True) -
                    pt.min(d_magnitude, dim=3, keepdims=True))
            # use data density as weights
            density = self._init_weights(Y, crit='cosine')
            density = pt.nan_to_num_(density, neginf=pt.nan)
            density = (density - np.nanmin(density, axis=2, keepdims=True)) / \
                        (np.nanmax(density, axis=2, keepdims=True) -
                        np.nanmin(density, axis=2, keepdims=True))
            return density.unsqueeze(0) + W
        elif type == 'ones':
            # ones, restore VMF
            return pt.ones(d_magnitude.shape)
        else:
            raise ValueError('Unknown type of weights!')

    def _init_weights(self, Y, q=20, sigma=100, crit='euclidean'):
        """ Compute the initial weights of data based on gaussian kernel
            w_i = \sum_j exp(-d(x_i, x_j)^2 / sigma), where j is the set of
            q-nearest neighbours of i. sigma is a positive scalar

        Args:
            Y: the initialzed data, shape (n_subj, N, P)
            q: the number of nearest neighbours. Defalut q = 20
            sigma: a positive scalar. Default sigma = 5

        Returns:
            the initial weights associated with each data point.
            shape (n_subj, 1, P)
        """
        data = pt.transpose(Y, 1, 2) # the tensor shape (subj, P, N)
        if crit == 'euclidean':
            dist = pt.cdist(data, data, p=2)
            W, idx = pt.topk(dist, q, dim=1, largest=False)
            W = pt.sum(pt.exp(-W ** 2 / sigma), dim=1, keepdim=True)
        elif crit == 'cosine':
            data = data / pt.sqrt(pt.sum(data ** 2, dim=2, keepdim=True))
            W, idx = pt.topk(pt.nan_to_num_(pt.matmul(data, pt.transpose(data, 1, 2)), nan=-pt.inf),
                             q+1, dim=1, largest=True)
            W = W[:, 1:, :]
            W = pt.mean(W, dim=1, keepdim=True)
        else:
            raise NameError('The input criterion must be either euclidean or cosine!')

        return W


class MixGaussianGamma(EmissionModel):
    """ Mixture of Gaussians with signal strength (fit gamma distribution)
        for each voxel. Scaling factor on the signal and a fixed noise variance
    """
    def __init__(self, K=4, N=10, P=20, data=None, params=None):
        """ Constructor for the Gaussian Mixture with gamma signal class

        Args:
            K (int): number of parcels
            N (int): number of tasks
            P (int): number of brain voxels
            data (np.array or pt.tensor): the data matrix, shape (n_subj, N, P)
            params (dict): the parameters of the model
        """
        super().__init__(K, N, P, data)
        self.random_params()
        self.set_param_list(['V', 'sigma2', 'alpha', 'beta'])
        self.name = 'GMM_gamma'
        if params is not None:
            self.set_params(params)
        self.tmp_list=['Y','YY','s','s2','rss','logs']


    def initialize(self, data, X=None):
        """ Stores the data in emission model itself
            Calculates sufficient stats on the data that does not depend on u,
            and allocates memory for the sufficient stats that does.
        """
        super().initialize(data,X=X)
        self.YY = self.Y ** 2
        self.s = pt.empty((self.num_subj, self.K, self.P))
        self.s2 = pt.empty((self.num_subj, self.K, self.P))
        self.rss = pt.empty((self.num_subj, self.K, self.P))
        self.logs = pt.empty((self.num_subj, self.K, self.P))

    def random_params(self):
        """ In this mixture gaussians, the parameters are parcel-specific mean V_k
            and variance. Here, we assume the variance is equal across different parcels.
            Therefore, there are total k+1 parameters in this mixture model
            set the initial random parameters for gaussian mixture
        """
        V = pt.randn(self.N, self.K)
        # standardise V to unit length
        V = V - V.mean(dim=0)
        self.V = V / pt.sqrt(pt.sum(V ** 2, dim=0))
        self.sigma2 = pt.tensor(np.exp(np.random.normal(0, 0.3)), dtype=pt.get_default_dtype())
        self.alpha = pt.tensor(1, dtype=pt.get_default_dtype())
        self.beta = pt.tensor(1, dtype=pt.get_default_dtype())

    def Estep(self, signal=None, sub=None):
        """ Estep: Returns log p(Y, s|U) for each value of U, up to a constant
            Collects the sufficient statistics for the M-step

        Args:
            signal: the signal strength for each subject, shape (nSubject, 1, P)
            sub: specify which subject to optimize

        Returns:
            LL (pt.tensor): the expected log likelihood for emission model,
                shape (nSubject * K * P)
        """
        n_subj = self.Y.shape[0]
        if sub is None:
            sub = range(n_subj)

        LL = pt.empty((n_subj, self.K, self.P))
        uVVu = pt.sum(self.V ** 2, dim=0)  # This is u.T V.T V u for each u

        for i in sub:
            YV = pt.mm(self.Y[i, :, :].T, self.V)
            # Importance sampling from p(s_i|y_i, u_i). First try sample from uniformed distribution
            # plt.figure()
            if signal is None:
                for k in range(self.K):
                    for p in range(self.P):
                        # Here try to sampling the posterior of p(s_i|y_i, u_i) for each
                        # given y_i and u_i(k)
                        x = pt.tensor(np.sort(np.random.uniform(0, 10, 1000)), dtype=pt.get_default_dtype())
                        loglike = - 0.5 * (1 / self.sigma2) * (-2 * YV[p, k] * x + uVVu[k] * x ** 2) + \
                                  (self.alpha - 1) * log(x) - self.beta * x
                        # This is the posterior prob distribution of p(s_i|y_i,u_i(k))
                        post = pt.exp(loglike) / pt.sum(pt.exp(loglike))
                        self.s[i, k, p] = pt.sum(x * post)
                        self.s2[i, k, p] = pt.sum(x**2 * post)
                        self.logs[i, k, p] = pt.sum(pt.log(x) * post)
                        # plt.plot(x, post)
                    # plt.show()

                self.s[i][self.s[i] < 0] = 0  # set all to non-negative
                self.s2[i][self.s2[i] < 0] = 0  # set all to non-negative
            else:
                self.s = signal.unsqueeze(1).repeat(1, self.K, 1)
                self.s2 = signal.unsqueeze(1).repeat(1, self.K, 1)**2
                self.logs = pt.log(signal.unsqueeze(1).repeat(1, self.K, 1))

            self.rss[i, :, :] = pt.sum(self.YY[i, :, :], dim=0) - 2 * YV.T * self.s[i, :, :] + \
                                self.s2[i, :, :] * uVVu.reshape((self.K, 1))
            # self.rss[i, :, :] = np.sum(self.YY[i, :, :], axis=0) - np.diag(np.dot(2*YV, self.s[i, :, :])) + \
            #                     np.dot(VV, self.s2[i, :, :])
            # the log likelihood for emission model (GMM in this case)
            LL[i, :, :] = -0.5 * self.N * (pt.log(2*self.PI)
                          + pt.log(self.sigma2)) - 0.5 * (1 / self.sigma2) * self.rss[i, :, :] \
                          + self.alpha * pt.log(self.beta) - pt.special.gammaln(self.alpha) \
                          + (self.alpha - 1) * self.logs[i, :, :] - self.beta * self.s[i, :, :]

        return LL

    def Mstep(self, U_hat):
        """ Performs the M-step on a specific U-hat. U_hat = E[u_i ^(k), s_i]
            In this emission model, the parameters need to be updated
            are V, sigma2, alpha, and beta

        Args:
            U_hat: The expected emission log likelihood
        """
        # SU = self.s * U_hat
        YUs = pt.zeros((self.N, self.K))
        US = pt.zeros((self.K, self.P))
        US2 = pt.zeros((self.K, self.P))
        UlogS = pt.zeros((self.K, self.P))
        ERSS = pt.zeros((self.num_subj, self.K, self.P))

        for i in range(self.num_subj):
            YV = pt.mm(self.Y[i, :, :].T, self.V)
            YUs = YUs + pt.mm(self.Y[i, :, :], (U_hat[i, :, :] * self.s[i, :, :]).T)
            US = US + U_hat[i, :, :] * self.s[i, :, :]
            US2 = US2 + U_hat[i, :, :] * self.s2[i, :, :]
            UlogS = UlogS + U_hat[i, :, :] * self.logs[i, :, :]
            ERSS[i, :, :] = pt.sum(self.YY[i, :, :], dim=0) - 2 * YV.T * U_hat[i, :, :] * self.s[i, :, :] + \
                            U_hat[i, :, :] * self.s2[i, :, :] * pt.sum(self.V ** 2, dim=0).reshape((self.K, 1))
            # ERSS[i, :, :] = np.sum(self.YY[i, :, :], axis=0) - np.diag(np.dot(2*YV, U_hat[i, :, :]*self.s[i, :, :])) + \
            #                     np.dot(self.V.T @ self.V, U_hat[i, :, :]*self.s2[i, :, :])

        # 1. Updating the V
        # rss = np.sum(self.YY, axis=1).reshape(self.num_subj, -1, self.P) \
        # - 2*np.transpose(np.dot(np.transpose(self.Y, (0, 2, 1)), self.V), (0,2,1))*U_hat*self.s + \
        # U_hat * self.s**2 * np.sum(self.V ** 2, axis=0).reshape((self.K, 1))
        self.V = YUs / pt.sum(US2, dim=1)

        # 2. Updating the sigma squared.
        # Here we update the v_k, which is sum_i(<Uhat(k), s_i>,*Y_i) / sum_i(Uhat(k), s_i^2)
        self.sigma2 = pt.sum(ERSS) / (self.N * self.P * self.num_subj)

        # 3. Updating the alpha
        self.alpha = 0.5 / (pt.log(pt.sum(US) / (self.num_subj * self.P)) - pt.sum(UlogS) / (self.num_subj * self.P))
        # self.alpha = self.alpha / self.beta  # first moment

        # 4. Updating the beta
        self.beta = (self.P * self.num_subj * self.alpha) / pt.sum(US)
        # self.beta = (self.alpha+1)*self.alpha / self.beta**2  # second moment

    def sample(self, U, signal=None, return_signal=False):
        """ Generate random data given this emission model and parameters

        Args:
            U: The prior arrangement U from the arrangement model
            V: Given the initial V. If None, then randomly generate
            return_signal: If true, return both data and signal; Otherwise, only data

        Returns:
            Y: The generated data with a shape of (num_subj, N, P)
            signal: The generated signal with a shape of (num_subj, P)
        """
        if type(U) is np.ndarray:
            U = pt.tensor(U, dtype=pt.int)
        elif type(U) is pt.Tensor:
            U = U.int()
        else:
            raise ValueError('The given U must be numpy ndarray or torch Tensor!')

        num_subj = U.shape[0]
        Y = pt.empty((num_subj, self.N, self.P))

        if signal is not None:
            np.testing.assert_equal(signal.shape, (num_subj, self.P),
                                    err_msg='The given signal must with a shape of (num_subj, P)')
        else:
            signal = pt.distributions.gamma.Gamma(self.alpha, self.beta).sample((num_subj, self.P))

        for s in range(num_subj):
            # Draw the signal strength for each node from a Gamma distribution
            Y[s, :, :] = self.V[:, U[s, :].long()] * signal[s, :]
            # And add noise of variance 1
            Y[s, :, :] = Y[s, :, :] + pt.normal(0, np.sqrt(self.sigma2), (self.N, self.P))

        if return_signal:
            return Y, signal
        else:
            return Y


####################################################################
## Belows are the helper functions for the emission models        ##
####################################################################
def loglik2prob(loglik, dim=0):
    """Safe transformation and normalization of
    logliklihood

    Args:
        loglik (ndarray): Log likelihood (not normalized)
        axis (int): Number of axis (or axes), along which the probability is being standardized
    Returns:
        prob (ndarray): Probability
    """
    if dim==0:
        ml, _ = pt.max(loglik, dim=0)
        loglik = loglik - ml + 10
        prob = np.exp(loglik)
        prob = prob / pt.sum(prob, dim=0)
    else:
        a = pt.tensor(loglik.shape)
        a[dim] = 1  # Insert singleton dimension
        ml, _ = pt.max(loglik, dim=0)
        loglik = loglik - ml.reshape(a) + 10
        prob = pt.exp(loglik)
        prob = prob/pt.sum(prob, dim=1).reshape(a)
    return prob


def bessel_function(self, order, kappa):
    """ The modified bessel function of the first kind of real order

    Args:
        order: the real order
        kappa: the input value

    Returns:
        res: The values of modified bessel function
    """
    # res = np.empty(kappa.shape)
    res = special.iv(order, kappa)
    return res

def log_bessel_function(order, kappa):
    """ The log of modified bessel function of the first kind of real order

    Args:
        order: the real order
        kappa: the input value

    Returns:
         The values of log of modified bessel function
    """
    PI = pt.tensor(pt.pi, dtype=pt.get_default_dtype())
    frac = kappa / order
    square = 1 + frac**2
    root = sqrt(square)
    eta = root + log(frac) - log(1 + root)
    approx = - log(sqrt(2 * PI * order)) + order * eta - 0.25*log(square)
    
    # Convert result to pytorch default dtype
    return approx.to(pt.get_default_dtype())