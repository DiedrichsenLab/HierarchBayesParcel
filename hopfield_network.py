#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 10/14/2021


Author: DZHI
'''

# Example Models
import os # to handle path information
import sys
sys.path.insert(0, "D:/python_workspace/")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.mixture import GaussianMixture
import scipy.stats as spst
import scipy as sp
from simulate import PottsModel, PottsModelGrid


def eucl_distance(coord):
    """
    Calculates euclediand distances over some cooordinates
    Args:
        coord (ndarray)
            Nx3 array of x,y,z coordinates
    Returns:
        dist (ndarray)
            NxN array pf distances
    """
    num_points = coord.shape[0]
    D = np.zeros((num_points,num_points))
    for i in range(3):
        D = D + (coord[:,i].reshape(-1,1)-coord[:,i])**2
    return np.sqrt(D)


def get_best_distribution(data):
    dist_names = ["norm", "exponnorm", "weibull_max", "gamma", "genextreme", "vonmises"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(spst, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = spst.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = " + str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: " + str(best_dist))
    print("Best p value: " + str(best_p))
    print("Parameters for the best fit: " + str(params[best_dist]))

    return best_dist, best_p, params[best_dist]


class HopfieldModelGrid(PottsModel):
    """
        Hopfield network model (fully connected Potts Model) defined on a regular grid
    """
    def __init__(self, K=3, width=5, height=5, r=2, k=0.5):
        self.width = width
        self.height = height
        self.dim = (height, width)
        self.r = r
        self.k = k
        # Get the grid and neighbourhood relationship
        W = self.define_grid()
        super().__init__(K, W)

    def define_grid(self):
        """
        Makes a connectivity matrix and a mapping matrix for a simple rectangular grid
        """
        XX, YY = np.meshgrid(range(self.width),range(self.height))
        self.xx = XX.reshape(-1)
        self.yy = YY.reshape(-1)
        self.Dist = np.sqrt((self.xx - self.xx.reshape((-1,1)))**2 + (self.yy - self.yy.reshape((-1,1)))**2)
        W = rbf_kernel(np.row_stack((self.xx, self.yy)).transpose(), gamma=self.k)  # Nearest neighbour connectivity within r
        # W = np.exp(-self.k * np.where((self.Dist > self.r) | (self.Dist == 0), np.nan, self.Dist))
        W = np.where((self.Dist > self.r) | (self.Dist == 0), np.nan, W)
        return W

    def estimation_params(self, data, labels):
        '''
        This function is used to estimate the probability distribution parameters
        by the given shape

        :param data: The functional profile, with shape [N_samples, n_conditions]
        :param labels: The given parcellations u_i
        :return: the parameters of gamma distribution
        '''
        all_param = []
        mean = data.mean(axis=1)
        data = data - mean[:, np.newaxis]
        if labels is None:
            U = np.random.choice(self.K, self.P)  # start with random parcellation
        else:
            U = labels

        for i in np.unique(U):
            this_data = data[U == i, :]
            if this_data.shape[0] == 0:
                this_prob = 0
            else:
                mag = np.linalg.norm(this_data, axis=1)

                param = spst.gamma.fit(mag[~np.isnan(mag)])
                all_param.append(param)

        return all_param

    def get_pdf(self, params, x):
        '''
        The function that returns the value from multiple probability distribution
        functions by different parameters
        :param params: With a shape of [N,m], where N represents the number of probability distributions
                       m is the number of parameters
        :param x: The data x
        :return: With a shape of N * 1, where N is the number of the probability density
        '''
        if params is None:
            return 1
        else:
            distribution = []
            for thetas in params:
                prob = spst.gamma.pdf(np.linalg.norm(x), a=thetas[0], loc=thetas[1], scale=thetas[2])
                distribution = np.append(distribution, prob)

            distribution = distribution/np.sum(distribution)
            return distribution

    def cond_prob(self, U, node, prior=False):
        """
        Returns the conditional probabity vector for node x, given U
        :param U: The current parcel arrangement, shape of N*1
        :param node: The index of node that wants to compute the conditional prob
        :param prior: set to True if there is a prior prob, default is Flase
        :return: The conditional probability P(x_i | x_j) where j belongs to
                 neighbouring nodes of i (Markov random field)
        """
        x = np.arange(self.K)
        ind = np.where(np.isnan(self.W[node, :]) == False)  # Find all the neighbors for node x (precompute!)
        nb_x = U[ind]  # Neighbors to node x
        same = np.equal(x, nb_x.reshape(-1, 1))
        loglik = self.theta_w * np.matmul(self.W[node, ind], same)
        loglik = loglik.reshape(-1)
        if prior:
            loglik = loglik + self.logMu[:, node]
        p = np.exp(loglik)
        p = p / np.sum(p)
        return p

    def sample_gibbs(self, U0=None, evidence=None, prior=False, iter=5, alpha=0.5, interval=None):
        """
        The main entry of the gibbs sampling
        :param U0: The initial parcel arrangement of the brain map
        :param evidence: The functional data covering each of the vertex
        :param prior: Th prior probability if applied
        :param iter: The max number of iterations
        :return: The result parcellation after # of iter gibbs sampling
        """
        # Get initial starting point if required
        U = np.zeros((iter+1, self.P))
        params = None
        if U0 is None:
            for p in range(self.P):
                U[0, p] = np.random.choice(self.K)
        else:
            U[0, :] = U0
        for i in range(iter):
            U[i+1, :] = U[i, :]  # Start the new sample from the old one
            liklihood = np.ones(self.P)
            if evidence is not None:
                # params = self.estimation_params(labels=U[i + 1, :], data=evidence)
                gmm = GaussianMixture(n_components=self.K, max_iter=1000, tol=1e-6, random_state=0).fit(evidence)
                liklihood = gmm.predict_proba(evidence)

            for p in range(self.P):
                prob = self.cond_prob(U[i+1, :], p, prior=prior)
                # liklihood = self.get_pdf(params, evidence[p])
                prob = np.multiply((1 - alpha) * prob, alpha * liklihood[p])
                prob = prob / np.sum(prob)
                U[i+1, p] = np.random.choice(self.K, p=prob)

        if interval is None:
            return U
        else:
            return U[np.append(0, np.arange(1, iter+1, interval)), :]

    def plot_maps(self, Y, cmap='tab20', vmin=0, vmax=19, grid=None):
        """
        Plots a set of map samples as an image grid, N X P
        :param Y: The data on the y-axis
        :param cmap: color map that being chosen
        :param vmin: the vmin parameter passed to imshow function
        :param vmax: the vmax parameter passed to imshow function
        :param grid: if None, there will be no grid applied to the plot
        :return: no return, show the plot
        """
        if Y.ndim == 1:
            ax = plt.imshow(Y.reshape(self.dim),cmap=cmap,interpolation='nearest',vmin=vmin,vmax=vmax)
            ax.axes.yaxis.set_visible(False)
            ax.axes.xaxis.set_visible(False)
        else:
            N, P = Y.shape
            if grid is None:
                grid = np.zeros((2,),np.int32)
                grid[0] = np.ceil(np.sqrt(N))
                grid[1] = np.ceil(N/grid[0])
            for n in range(N):
                ax = plt.subplot(grid[0], grid[1], n+1)
                if n % grid[1] == 0:
                    ax.set_ylabel('Sub %d' % np.int(n/grid[1] + 1))
                    ax.set_xlabel('Prior')
                elif n % grid[1] == grid[1] - 1:
                    ax.set_xlabel('original parcellation')
                # else:
                #     ax.set_xlabel('iter = %d' % )
                ax.imshow(Y[n,:].reshape(self.dim), cmap=cmap, interpolation='nearest',vmin=vmin,vmax=vmax)
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)


if __name__ == '__main__':
    # Generate a Pottsmodel on a rectangular 30x30 Grid
    M = PottsModelGrid(5, 30, 30)
    # Define the parcellation + Prior Probability
    M.define_mu(200)
    # Show the clustering
    plt.figure(figsize=(2, 2))
    cluster = np.argmax(M.mu, axis=0)
    M.plot_maps(cluster)
    plt.show()

    # Show the prior probabilities for the different parcels
    plt.figure(figsize=(10, 2))
    M.plot_maps(M.mu, cmap='jet', vmax=1, grid=[1, 5])
    plt.show()

    # Generate a data set for a specific experiment
    # We can now generate a data set for a specific experiment.
    N = 5
    sub_par = M.generate_subjects(num_subj=N)
    [Y, param] = M.generate_emission(sub_par)

    # Generate a hopfield model on a rectangular 30x30 Grid
    # cluster k, grid height, weight, mrf max neighbour distance, rbf kernel parameter
    H = HopfieldModelGrid(5, 30, 30, 2, 0.5)
    # Define the parcellation + Prior Probability
    H.define_mu(200)
    cluster = np.argmax(H.mu, axis=0)

    # Gibbs sampling
    # Demonstrate a single Gibbs sample, starting from a random start (drawn independently from mu)
    H.theta_w = 5  # inverse temperature parameter
    toPlot = np.zeros((1, H.P))
    for i in range(N):
        Us = H.sample_gibbs(U0=cluster, prior=True, iter=50, evidence=Y[i, :, :].transpose(), alpha=0.5, interval=5)
        Us = np.vstack((Us, sub_par[i]))
        toPlot = np.vstack((toPlot, Us))

    toPlot = np.delete(toPlot, 0, axis=0)
    plt.figure(figsize=(toPlot.shape[0]/N, N))
    H.plot_maps(toPlot, cmap='tab20', grid=[N, np.int(toPlot.shape[0]/N)])
    plt.ylabel('sujects per row')
    plt.xlabel('Prior -> iterations of gibbs sampling -> original parcellation')
    plt.show()
    print('Done')

