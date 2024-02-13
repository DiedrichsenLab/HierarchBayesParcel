#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/14/2021
Emission models class

Author: dzhi, jdiedrichsen
"""
import numpy as np
import torch as pt
import pandas as pd
import pickle
from scipy import stats, special
from torch import log, exp, sqrt
from HierarchBayesParcel.model import Model
from HierarchBayesParcel.depreciated.AIS_test import rejection_sampling
import HierarchBayesParcel.arrangements as ar

class EmissionModel(Model):
    """ Abstract class for emission models
    """
    def __init__(self, K=4, N=10, P=20, num_subj=None, X=None):
        """ Abstract constructor of emission models
        Args:
            K: the number of clusters
            N: the number of observations if given
            P: the number of brain locations
            num_subj: Number of subjects, if important for parameters (e.g. subject specific)
            X: the design matrix of observations,
               shape of (N, M) tensor (overwrits N if given)
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
        if num_subj is not None:    
            self.num_subj = num_subj
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
    def __init__(self, K=4, N=10, P=20, num_subj=None, X=None,
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
        super().__init__(K, N, P, num_subj, X)
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

        if 'V' in self.param_list:
            self.V = pt.matmul(regressX, pt.sum(YU, dim=0)/this_U_hat.sum(dim=(0, 2)))
            if self.std_V:
                self.V = self.V / pt.sqrt(pt.sum(self.V ** 2, dim=0))

        # 2. Updating sigma2 (rss is calculated using updated V)
        if 'sigma2' in self.param_list:
            YV = pt.matmul(pt.matmul(self.X, self.V).T, self.Y)
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
    """ Mixture of von Mises-Fisher distribution emission model
    """
    def __init__(self, K=4, N=10, P=20, 
                num_subj=None,
                X=None,
                part_vec=None,
                params=None,
                uniform_kappa=None,
                parcel_specific_kappa=False,
                subject_specific_kappa=False,
                subjects_equal_weight=False
                ):
        """ Constructor for the vmf mixture emission model
        Args:
            K (int): the number of clusters
            N (int): the number of observations 
            P (int): the number of voxels
            num_subj (int): number of subjects 
            X (ndarray or tensor): N x M design matrix for task conditions
            part_vec (ndarray or tensor): M-Vector indicating the number of the
                      data partition (repetition).
                      Expample = [1,2,3,1,2,3,...] None: no data partition
            params: if None, no parameters to pass in.
                Otherwise take the passing parameters as the model params
            uniform_kappa (bool): Defaults to True
            parcel_specific_kappa (bool): Defaults to False,
            subject_specific_kappa (bool): Defaults to False,
            subjects_equal_weight (bool): if False, mstep is average across all voxels in region
                if True, first averages within subejct across voxels - then across subjects (equal weight for each subejct)
        """
        # Set flags for M-step 
        self.parcel_specific_kappa = parcel_specific_kappa
        self.subject_specific_kappa = subject_specific_kappa
        self.subjects_equal_weight = subjects_equal_weight
        if uniform_kappa is not None:
            warn.warn('setting uniform_kappa is depreciated - set parcel_specific_kappa / subject-specific kappa instead',warn.DeprecationWarning)
            if uniform_kappa == False:
                self.parcel_specific_kappa = True

        if part_vec is not None:
            if isinstance(part_vec,(np.ndarray,pt.Tensor)):
                self.part_vec = pt.tensor(part_vec, dtype=pt.int)
            else:
                raise ValueError('Part_vec must be numpy ndarray or torch Tensor')
        else:
            self.part_vec = None

        super().__init__(K, N, P, num_subj, X)
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

        if self.parcel_specific_kappa:
            self.kappa = pt.distributions.uniform.Uniform(10, 150).sample((self.K, ))
        elif self.subject_specific_kappa:
            self.kappa = pt.distributions.uniform.Uniform(10, 150).sample((self.num_subj,))
        else:
            self.kappa = pt.distributions.uniform.Uniform(10, 150).sample()

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
        # Normalization constant for the von Mises-Fisher distribution for current Kappa
        logCnK = (self.M/2 - 1)*log(self.kappa) - (self.M/2)*log(2*PI) - \
                 log_bessel_function(self.M/2 - 1, self.kappa)

        if self.parcel_specific_kappa:
            LL = logCnK.unsqueeze(0).unsqueeze(2) * self.num_part + self.kappa.unsqueeze(1) * YV
        elif self.subject_specific_kappa:
            LL = logCnK.unsqueeze(1).unsqueeze(2) * self.num_part + self.kappa.unsqueeze(1).unsqueeze(2) * YV
        else:
            LL = logCnK * self.num_part + self.kappa * YV

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
        JU = self.num_part * U_hat
        # Calculate YU = \sum_i\sum_k<u_i^k>y_i and UU = \sum_i\sum_k<u_i^k>
        YU = pt.matmul(pt.nan_to_num(self.Y), pt.transpose(U_hat, 1, 2))

        # If the subejcts are weighted differently - just sum voxels per regions across subjects 
        r_norm2 = pt.sum(YU ** 2, dim=1, keepdim=True)
        r_norm2[r_norm2 == 0] = pt.nan # Avoid division by zero
        
        # 1. Updating the V_k, which is || sum_i(Uhat(k)*Y_i) / sum_i(Uhat(k)) ||
        if 'V' in self.param_list:
            if self.subjects_equal_weight:
                self.V = YU / pt.sqrt(r_norm2)
                self.V = pt.nanmean(self.V,dim=0)
                self.V = self.V / pt.sqrt(pt.sum(self.V ** 2, dim=0))
            else:
                YU_tot = pt.sum(YU, dim=0)
                r_norm2_tot = pt.sum(YU_tot**2, dim=0)
                r_norm2_tot[r_norm2_tot == 0] = pt.nan # Avoid division by zero
                self.V = YU_tot / pt.sqrt(r_norm2_tot)

        # 2. Updating kappa, kappa_k = (r_bar*N - r_bar^3)/(1-r_bar^2),
        # where r_bar = ||V_k||/N*Uhat
        if 'kappa' in self.param_list:
            if self.subjects_equal_weight:
                pass
            else: 
                if self.parcel_specific_kappa and not self.subject_specific_kappa:
                    r_bar = pt.sqrt(r_norm2_tot) / pt.sum(JU, dim=[0,2])
                    r_bar[r_bar > 0.95] = 0.95
                    r_bar[r_bar < 0.05] = 0.05
                elif self.subject_specific_kappa and not self.parcel_specific_kappa:
                    r_bar = pt.sqrt(r_norm2).sum(dim=[1,2]) / pt.sum(JU, dim=[1,2])
                elif not self.subject_specific_kappa and not self.parcel_specific_kappa:
                    r_bar = pt.sqrt(r_norm2).sum() / self.num_part.sum()
                else:
                    raise ValueError('Subject & Regions specific kappa is not implemented yet')    
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
                if self.parcel_specific_kappa:
                    y = pt.tensor(random_VMF(self.V[:, this_par].cpu().numpy(),
                                             self.kappa[this_par].cpu().numpy(),
                                             int(counts[i] * num_parts)),
                                  dtype=pt.get_default_dtype())
                elif self.subject_specific_kappa:
                    y = pt.tensor(random_VMF(self.V[:, this_par].cpu().numpy(),
                                             self.kappa[s].cpu().numpy(),
                                             int(counts[i] * num_parts)),
                                  dtype=pt.get_default_dtype())
                else: 
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

def random_VMF(mu, kappa, size=None):
    """ von Mises-Fisher distribution sampler with
        mean direction mu and co nc en tr at io n kappa .
        Source: https://hal.science/hal-04004568

    Args:
        mu: The mean direction vector
        kappa: The concentration parameter
        size: The number of the output samples

    Returns:
        The samples from the von Mises-Fisher distribution

    References:
        Carlos Pinzón, Kangsoo Jung. Fast Python sampler for the
        von Mises Fisher distribution. 2023. hal-04004568v2
    """
    # parse input parameters
    n = 1 if size is None else np.product(size)
    shape=() if size is None else tuple (np.ravel(size))
    mu = np.asarray(mu)
    mu = mu/np.linalg.norm(mu)
    (d, ) = mu.shape
    # z component : radial samples pe rp en dic ul ar to mu
    z = np.random.normal(0, 1, (n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    z = z - (z @ mu[:, None]) * mu[None, :]
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    # sample angles ( in cos and sin form )
    cos = _random_VMF_cos(d, kappa, n)
    sin = np.sqrt(1 - cos ** 2)
    # combine angles with the z component
    x = z * sin[:, None] + cos[:, None] * mu[None , :]

    return x.reshape((*shape, d))

def _random_VMF_cos(d: int, kappa: float, n: int):
    """ Generate n iid samples t with density function given by
        p(t) = someConstant * (1-t**2) **((d-2)/2) * exp ( kappa *t)

    Args:
        d: The dimension of the samples
        kappa: The concentration parameter
        n: The number of the samples

    Returns:
        The samples of angles in cos form
    """
    b = (d-1) / (2*kappa + (4*kappa ** 2 + (d-1)**2)**0.5)
    x0 = (1-b) / (1+b)
    c = kappa * x0 + (d - 1) * np.log(1 - x0 ** 2)
    found = 0
    out = []
    while found < n:
        m = min(n, int((n - found) * 1.5))
        z = np.random.beta((d - 1) / 2, (d - 1)/2, size=m)
        t = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        test = kappa * t + (d - 1) * np.log(1 - x0 * t) - c
        accept = test >= -np.random.exponential(size=m)
        out.append(t[accept])
        found += len(out[-1])

    return np.concatenate(out)[:n]

def load_emission_params(fname, param_name, index=None, 
                         device=None):
    """ Loads parameters from a list of emission models 
        of a pre-trained model.

    Args:
        fname (str): File name of pre-trained model
        param_name (str): Name of the parameter to load
        index (int): Index of the model to load. If None,
            loads the model with the highest log-likelihood
            by default.
        device (str): Device to load the model to. Current
            support 'cuda' and 'cpu'.

    Returns:
        params (list): a list of emission parameters from
            the emission models
        info_reduced (pandas.Dataframe): Data Frame with
            necessary information
    """
    info = pd.read_csv(fname + '.tsv', sep='\t')
    with open(fname + '.pickle', 'rb') as file:
        models = pickle.load(file)

    if index is None:
        index = info.loglik.argmax()

    select_model = models[index]
    if device is not None:
        select_model.move_to(device)

    info_reduced = info.iloc[index]
    params = []
    for em_model in select_model.emissions:
        assert param_name in em_model.param_list, \
            f'{param_name} is not in the param_list.'
        params.append(vars(em_model)[param_name])

    return params, info_reduced

def build_emission_model(K, atlas, emission, x_matrix, part_vec, V=None,
                         em_params={}):
    """ Builds an arrangment model based on the specification

    Args:
        K (int): number of voxels
        atlas (object): the atlas object for the arrangement model
        emission (str): the emission model type
        x_matrix (ndarray): the design matrix associated with an
            emission model
        part_vec (ndarray): A partition vector of the data
        V (torch.tensor or numpy.ndarray): the mean direction
            parameter for VMF emission model or the mean response
            parameter for GMM emission model. If None, the parameter
            for current building emission model will be randomly
            initialized. If V is given, the parameter 'V' is fixed
            throughout the modeling learning.
        em_params (dictionary): Parameters passed to the emission model
            constructor


    Returns:
        em_model (object): the emission model object
    """
    if emission == 'VMF':
        em_model = MixVMF(K=K, P=atlas.P, X=x_matrix,
                          part_vec=part_vec,
                          **em_params)
    elif emission == 'GMM':
        em_model = MixGaussian(K=K, P=atlas.P, X=x_matrix,
                               **em_params
                            )
    else:
        raise ((NameError(f'unknown emission model:{emission}')))
    
    # If V is given, then remove it from parameter list and only 
    # update the kappa
    if V is not None:
        em_model.V = V
        new_param_list = em_model.param_list.copy()
        new_param_list.remove('V')
        em_model.set_param_list(new_param_list)
        
    return em_model
