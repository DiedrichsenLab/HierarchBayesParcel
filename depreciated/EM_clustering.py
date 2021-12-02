#####################################################################################
# EM algorithm for clustering Mixture Model and visualization
#
# Date: Nov. 25, 2018
# Author: Da Zhi
#####################################################################################
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
from scipy.stats import multivariate_normal

DEBUG = True  # global debugging flag


# -------------------------------
# Global debugging flag - DEBUG
# -------------------------------
def debug(*args, **kwargs):
    global DEBUG
    if DEBUG:
        print(*args, **kwargs)


def sample_GMM(dims=2, n_sample=500, components=5, sample_assignment=None, normalize=True):
    '''
    Generate random samples from N gaussian mixtures
    :param dims: the dimensions of returned data sample, default = 2
    :param n_sample: the number of returned data sample
    :param components: the number of GMM used to sample data
    :param sample_assignment: the weights of each GMM of the
           total number of samples

    :return: data samples from n GMMs, shape = [n_sample, dims]
    '''
    data = np.empty([1, dims])
    if sample_assignment is None:
        a = np.random.dirichlet(np.ones(components))
        nums = np.rint(n_sample * a[:-1])
        nums = np.append(nums, n_sample-nums.sum()).astype('int')
    else:
        nums = np.rint(n_sample * sample_assignment).astype('int')

    mu, cov, data_classified = [], [], []
    for i in range(components):
        this_mu = np.random.uniform(-1, 1, dims)
        A = np.random.rand(dims, dims)
        this_cov = np.mat(np.dot(A, A.transpose()))
        this_data = np.random.multivariate_normal(this_mu, this_cov, nums[i])
        data = np.vstack((data, this_data))
        mu.append(this_mu)
        cov.append(this_cov)
        data_classified.append(this_data)

    data = np.delete(data, 0, axis=0)
    if normalize:
        data = 2. * (data - np.min(data)) / np.ptp(data) - 1
        return np.asarray(mu), np.asarray(cov), data, data_classified
    else:
        return np.asarray(mu), np.asarray(cov), data, data_classified


# -------------------------------------------------------------------
# Calculate the normal of probability of Xn given mu and covariance
# for the kth model using function 'multivariate_normal', this
# function is similar to 'mvnpdf' in MATLAB
# Input: Y - data points
#        mu_k - the mu array of k models
#        cov_k - the covariance of k models
# Return: norm.pdf(Y) (1-D array)
# -------------------------------------------------------------------
def phi(data, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(data)


# ---------------------------------------------------------------------------------
# E - Step：Calculate the expectation of each of the model
# Input Parameters: Y - the matrix of data points with shape (400, 1)
#                   mu - the mean of each clusters
#                   cov - the covariance
#                   pi - cluster probability distribution of each point
# Return: gamma - the probability of data sample is from the k clusters
#         loglikelihood - the loglikelihood of current iteration
# ---------------------------------------------------------------------------------
def Expectation(Y, mu, cov, pi):
    """
    E-step: calculate the expectation of each of the gaussian model

    :param Y: the matrix of data points with shape ()
    :param mu: the mean of each clusters
    :param cov: the covariance
    :param pi: cluster probability distribution of each point
    :return: gamma - the probability of data sample is from the k clusters
             loglikelihood - the log likelihood of current iteration
    """
    # number of samples (data points)
    N = Y.shape[0]
    # number of models (clusters)
    K = pi.shape[0]
    loglikelihood = 0

    # The number of samples and mixture model are restricted equal to 1 to avoid different return types
    assert N > 1, "There should be more than one data sample"
    assert K > 1, "There should be more than one gaussian mixture model"

    # Initialize gamma, size of (400, k)
    gamma = np.mat(np.zeros((N, K)))

    # Calculate the probability of occurrence of all samples in each model,
    # row corresponding samples, column corresponding models
    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, k] = phi(Y, mu[k], cov[k])
    prob = np.mat(prob)

    # Calculate the gamma of each model to each sample
    for k in range(K):
        gamma[:, k] = pi[k] * prob[:, k]

    # --------------------------------------------------------
    # Compute the log likelihood in current E-step iteration
    # --------------------------------------------------------
    for i in range(N):
        sum1data = np.log(np.sum(gamma[i, :]))
        loglikelihood += sum1data
        gamma[i, :] /= np.sum(gamma[i, :])

    return gamma, loglikelihood


# -----------------------------------------------------------------------
# M - Step：iteration computation of the probability distribution
#           to maximize the expectation
# Input parameters: Y - input data points matrix
#                   gamma - the result from E-step
# Return: new mu, covariance, and probability distribution pi
# -----------------------------------------------------------------------
def maximization(Y, gamma):
    # get shape of input data matrix, k for initialization
    N, D = Y.shape
    K = gamma.shape[1]
    mu = np.zeros((K, D))
    cov = []
    pi = np.zeros(K)

    # Update mu, cov, and pi for each of the models (clusters)
    for k in range(K):
        Nk = np.sum(gamma[:, k])
        # update mu, get the mean of each of the column
        for d in range(D):
            mu[k, d] = np.sum(np.multiply(gamma[:, k], Y[:, d])) / Nk
        # updata covariance
        cov_k = np.mat(np.zeros((D, D)))
        for i in range(N):
            cov_k += gamma[i, k] * (Y[i] - mu[k]).T * (Y[i] - mu[k]) / Nk
        cov.append(cov_k)
        # update pi
        pi[k] = Nk / N
    cov = np.array(cov)
    return mu, cov, pi


# ---------------------------------------------
# Data pre-processing
# Scaling all point data within [0, 1]
# ---------------------------------------------
def scale_data(Y):
    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    debug("Data scaled.")
    return Y


# ---------------------------------------------------------------------
# Initialization the parameters of mixture model
# Input parameters: shape - the shape of data points (400, 2)
#                   K - the number of models (clusters)
# Return: mu - randomly generate, size of (k, 2)
#         covariance - k covariance matrices - identity
#         pi - the initial probability distribution - 1/k
# ---------------------------------------------------------------------
def init_params(shape, k):
    N, D = shape
    mu = np.random.rand(k, D)
    cov = np.array([np.eye(D)] * k)
    pi = np.array([1.0 / k] * k)
    debug("Parameters initialized.")
    debug("mu:", mu, "cov:", cov, "pi:", pi, sep="\n")
    return mu, cov, pi


# ------------------------------------------------------------------
# The main entry of the EM algorithm for mixture model
# Input: Y - the matrix of data input
#        k - the number of gaussian model (clusters)
#        time - the number of iteration
# Return: the final mu, cov, pi, and the array of loglikelihood
# ------------------------------------------------------------------
def MM_EM(datapoint, k, times):
    datapoint = scale_data(datapoint)
    mu, cov, pi = init_params(datapoint.shape, k)
    likelihoodarray = []
    for i in range(times):
        gamma, loglikelihood = Expectation(datapoint, mu, cov, pi)
        mu, cov, pi = maximization(datapoint, gamma)
        likelihoodarray.append(loglikelihood)

    debug("{sep} Result {sep}".format(sep="-" * 20))
    debug("mu:", mu, "cov:", cov, "pi:", pi, sep="\n")
    return mu, cov, pi, likelihoodarray


if __name__ == "__main__":
    DEBUG = True  # debugging mode flag

    # Load data from .mat file as array-like data
    # mat = spio.loadmat("mixtureData.mat")
    # Y = mat['Y']
    # print(Y.shape)
    # matY = np.matrix(Y, copy=True)

    # Generate random samples from N gaussian mixtures
    prior_mu, prior_cov, matY, Y_class = sample_GMM(dims=2, n_sample=100, components=3)
    matY = np.matrix(matY, copy=True)

    # the main entry of EM algorithm for mixture model, to change the k
    k = 3
    mu, cov, pi, loglikelihoods = MM_EM(matY, k, 100)

    # get the final gamma for clustering
    N = matY.shape[0]
    gamma, likelihood = Expectation(matY, mu, cov, pi)
    category = gamma.argmax(axis=1).flatten().tolist()[0]

    # Separating all data point into k clusters and store them
    category = np.asarray(category)

    # Plot results
    colors = plt.cm.rainbow(np.linspace(0, 1, k))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    for i in range(k):
        this_classes = Y_class[i]
        ax1.plot(this_classes[:, 0], this_classes[:, 1], 'o', color=colors[i])

    ax1.set_title('Random samples from GMMs')

    for i in range(k):
        this_classes = matY[category == i]
        ax2.plot(this_classes[:, 0], this_classes[:, 1], 'o', color=colors[i])

    ax2.set_title('EM algorithm results')
    plt.legend(loc="best")
    plt.show()

    plt.figure()
    plt.plot(loglikelihoods)
    plt.title("Log likelihood")
    plt.show()
