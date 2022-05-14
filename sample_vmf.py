#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/14/2021
Supplementary functions for sampling from von-Mises Fisher distribution
using code provided in git repo (References [1] and [2]).
Special thanks to the author of the blog [1], Daniel L. Whittenbury

Modified by: dzhi, to meet the generative framework use

References:
    [1] https://dlwhittenbury.github.io/ds-2-sampling-and-visualising-the-von-mises-fisher-distribution-in-p-dimensions.html
    [2] https://github.com/dlwhittenbury/von-Mises-Fisher-Sampling
"""
import numpy as np
from scipy.linalg import null_space
import numpy.matlib

fs = 16


def rand_uniform_hypersphere(N, p):
    """Generate random samples from the uniform distribution
    on the (p-1)-dimensional hypersphere
    Args:
        N: (int) Number of samples
        p: (int) The dimension of the generated samples on
           the (p-1)-dimensional hypersphere.
    Returns:
        random samples
    """
    if (p <= 0) or not isinstance(p, int):
        raise Exception("p must be a positive integer.")

    # Check N>0 and is an int
    if (N <= 0) or not isinstance(N, int):
        raise Exception("N must be a non-zero positive integer.")

    v = np.random.normal(0, 1, (N, p))

    #    for i in range(N):
    #        v[i,:] = v[i,:]/np.linalg.norm(v[i,:])

    v = np.divide(v, np.linalg.norm(v, axis=1, keepdims=True))

    return v


def rand_t_marginal(kappa, p, N=1):
    """Samples the marginal distribution of t using rejection
       sampling.
    Args:
        kappa: concentration parameter (float)
        p: The dimension of the generated samples on the
           (p-1)-dimensional hypersphere. (int)
        N: number of samples (int)
    Returns:
        samples: (N,1) samples of the marginal distribution of t
    """

    # Check kappa >= 0 is numeric
    if (kappa < 0) or (isinstance(kappa, float) and isinstance(kappa, int)):
        raise Exception("kappa must be a non-negative number.")

    if (p <= 0) or not isinstance(p, int):
        raise Exception("p must be a positive integer.")

    # Check N>0 and is an int
    if (N <= 0) or not isinstance(N, int):
        raise Exception("N must be a non-zero positive integer.")

    # Start of algorithm
    b = (p - 1.0) / (2.0 * kappa + np.sqrt(4.0 * kappa ** 2 + (p - 1.0) ** 2))
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (p - 1.0) * np.log(1.0 - x0 ** 2)

    samples = np.zeros((N, 1))

    # Loop over number of samples
    for i in range(N):
        while True:
            # Sample Beta distribution
            Z = np.random.beta((p - 1.0) / 2.0, (p - 1.0) / 2.0)

            # Sample Uniform distribution
            U = np.random.uniform(low=0.0, high=1.0)
            W = (1.0 - (1.0 + b) * Z) / (1.0 - (1.0 - b) * Z)

            # Check whether to accept or reject
            if kappa * W + (p - 1.0) * np.log(1.0 - x0 * W) - c >= np.log(U):
                samples[i] = W  # Accept sample
                break

    return samples


def rand_von_mises_fisher(mu, kappa, N=1):
    """Samples the von Mises-Fisher distribution with mean
       direction mu and concentration kappa.
    Args:
        mu: (p,1) mean direction. This should be a unit vector.
        kappa: concentration parameter (float)
        N: number of samples (int)
    Returns:
        samples: samples of the von Mises-Fisher distribution
        using mean direction mu and concentration kappa.
    """
    # Check that mu is a unit vector
    eps = 10 ** (-8)  # Precision
    norm_mu = np.linalg.norm(mu)
    if abs(norm_mu - 1.0) > eps:
        raise Exception("mu must be a unit vector.")

    # Check kappa >= 0 is numeric
    if (kappa < 0) or (isinstance(kappa, float) and isinstance(kappa, int)):
        raise Exception("kappa must be a non-negative number.")

    # Check N>0 and is an int
    if (N <= 0) or not isinstance(N, int):
        raise Exception("N must be a non-zero positive integer.")

    p = len(mu)
    mu = np.reshape(mu, (p, 1))
    samples = np.zeros((N, p))
    t = rand_t_marginal(kappa, p, N)  # Component in the direction of mu (Nx1)
    xi = rand_uniform_hypersphere(N, p - 1)  # Component orthogonal to mu (Nx(p-1))

    # von-Mises-Fisher samples Nxp
    samples[:, [0]] = t

    # Component orthogonal to mu (Nx(p-1))
    samples[:, 1:] = np.matlib.repmat(np.sqrt(1 - t ** 2), 1, p - 1) * xi

    # Rotation of samples to desired mu
    O = null_space(mu.T)
    R = np.concatenate((mu, O), axis=1)
    samples = np.dot(R, samples.T).T

    return samples
