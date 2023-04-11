#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/14/2021
Supplementary functions for sampling from von-Mises Fisher distribution
using code provided in git repo (References [1] and [2]).
Special thanks to the author of the blog [1], Daniel L. Whittenbury

Modified by: dzhi, for the generative framework use

References:
    [1] https://dlwhittenbury.github.io/ds-2-sampling-and-visualising
        -the-von-mises-fisher-distribution-in-p-dimensions.html
    [2] https://github.com/dlwhittenbury/von-Mises-Fisher-Sampling
"""
import numpy as np
import torch as pt

def _nullspace(A, rcond=None):
    """Compute an approximate basis for the nullspace of A. This
     is equivalent function in pyTorch to the scipy.linalg.null_space

    Args:
        A: (torch.tensor) A should be at most 2-D. A 1-D array
            with length
        rcond: (float) Cut-off ratio for small singular values
            of A. If None, the value is calculated using the
            machine precision of the data type

    Returns:
        nullspace: (torch.tensor) An array whose columns form a basis
         for the nullspace of A.
    """
    U, S, V = pt.linalg.svd(A)
    if rcond is None:
        rcondt = pt.finfo(S.dtype).eps * max(U.shape[0], V.shape[0])
    tolt = pt.max(S) * rcondt
    numt= pt.sum(S > tolt, dtype=int)
    nullspace = V[numt:,:].T

    return nullspace

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

    v = pt.randn(N, p)
    v = v / pt.norm(v, dim=1, keepdim=True)

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
    b = (p - 1.0) / (2.0 * kappa + pt.sqrt(4.0 * kappa ** 2 + (p - 1.0) ** 2))
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (p - 1.0) * pt.log(1.0 - x0 ** 2)

    samples = pt.zeros((N, 1))
    # Loop over number of samples
    for i in range(N):
        while True:
            # Sample Beta distribution
            Z = pt.distributions.beta.Beta((p - 1.0) / 2.0, (p - 1.0) / 2.0).sample()
            # Sample Uniform distribution
            U = pt.distributions.uniform.Uniform(0.0, 1.0).sample()
            W = (1.0 - (1.0 + b) * Z) / (1.0 - (1.0 - b) * Z)

            # Check whether to accept or reject
            if kappa * W + (p - 1.0) * pt.log(1.0 - x0 * W) - c >= pt.log(U):
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
    norm_mu = pt.linalg.norm(mu)
    if pt.abs(norm_mu - 1.0) > eps:
        raise Exception("mu must be a unit vector.")

    # Check kappa >= 0 is numeric
    if (kappa < 0) or (isinstance(kappa, float) and isinstance(kappa, int)):
        raise Exception("kappa must be a non-negative number.")

    # Check N>0 and is an int
    if (N <= 0) or not isinstance(N, int):
        raise Exception("N must be a non-zero positive integer.")

    p = len(mu)
    mu = pt.reshape(mu, (p, 1))
    samples = pt.zeros((N, p))
    t = rand_t_marginal(kappa, p, N)  # Component in the direction of mu (Nx1)
    xi = rand_uniform_hypersphere(N, p-1)  # Component orthogonal to mu (Nx(p-1))

    # von-Mises-Fisher samples Nxp
    samples[:, [0]] = t
    # Component orthogonal to mu (Nx(p-1))
    samples[:, 1:] = pt.sqrt(1 - t**2).expand(-1, p-1) * xi

    # Rotation of samples to desired mu
    O = _nullspace(mu.T)
    R = pt.cat((mu, O), dim=1)
    samples = pt.mm(R, samples.T).T

    return samples
