#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2/14/2022
Testing AIS (Annealed Importance Sampling)

Author: DZHI
"""
import torch as pt
import matplotlib.pyplot as plt
pt.set_default_dtype(pt.float64)
device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")

# def f_0(x):
#     """
#     Target distribution: \propto N(-5, 2)
#     """
#     return np.exp(-(x+5)**2/2/2)
#
#
# def f_n(x):
#     """
#     Proposed distribution: gaussian
#     """
#     return np.exp(-np.power(x - 0, 2.) / (2 * np.power(2, 2.)))


# def f_j(x, beta):
#     """
#     Intermediate distribution: interpolation between f_0 and f_n
#     """
#     return f_0(x)**beta * f_n(x)**(1-beta)


def f_transition(x, f, n_steps=10):
    """
    Transition distribution: T(x'|x) using n-steps Metropolis sampler
    """
    for t in range(n_steps):
        # Proposal
        x_prime = x + pt.distributions.normal.Normal(0, 1).sample()

        # Acceptance prob
        a = f(x_prime) / f(x)

        if pt.distributions.uniform.Uniform(0, 1).sample() < a:
            x = x_prime

        # # Acceptance prob
        # a = f(x_prime) / f(x)
        # if a >= 1:
        #     x = x_prime
        # else:
        #     if pt.distributions.uniform.Uniform(0, 1).sample() < a:
        #         x = x_prime
    return x


def annealed_importance_sampling(f_0, f_n, q_x, num_sample=1000, interval=50):
    """Annealed importance sampling from the proposal distribution q(x) to
       the target distribution p(x) using many intermediate transition probability
       distributions.
    Args:
        f_0: the target distribution function
        f_n: the proposal distribution function
        q_x: the proposal distribution q(x) sampler
        num_sample: the number of samples
        interval: the number of intermediate transition
                  distributions used from f_n to f_0
    Returns:
        the expectation of the target function
    """
    betas = pt.linspace(0, 1, interval)
    samples = pt.zeros(num_sample)
    weights = pt.zeros(num_sample)
    for t in range(num_sample):
        # Sample initial point from q(x)
        x = q_x.sample()
        w = 0
        for n in range(1, len(betas)):
            # Transition
            f_j = lambda i: f_0(i)**betas[n] * f_n(i)**(1-betas[n])
            f_j_1 = lambda i: f_0(i)**betas[n-1] * f_n(i)**(1-betas[n-1])
            x = f_transition(x, f_j, n_steps=5)

            # Compute weight in log space (log-sum):
            # w *= f_{n-1}(x_{n-1}) / f_n(x_{n-1})
            w += pt.log(f_j(x)) - pt.log(f_j_1(x))

        samples[t] = x
        weights[t] = pt.exp(w)  # Transform back using exp

    # Compute expectation
    first_mom = 1/pt.sum(weights) * pt.sum(weights * samples)
    second_mom = 1/pt.sum(weights) * pt.sum(weights * samples**2)

    return first_mom, second_mom


def rejection_sampling(p_x, q_x=None, sampling_range=None, num_sample=100, interval=50):
    """Perform rejection sampling from q(x) to p(x)
    Args:
        p_x: the target distribution (un-normalized)
        q_x: the proposal distribution (known pdf of torch distribution object)
        sampling_range: the range of the sampling
        num_sample: number of samples
        interval: the interval between 0 to 10 for finding the C if range is None
    Returns:
        expectation E(x) wrt. q(x)
    """
    # step 1. find constant c to make q(x) to cover target distribution by sampling
    if sampling_range is None:
        sampling_range = pt.linspace(0, 10, interval)

    if q_x is None:  # if q(x) not given, then use normal(0,1) as proposal q(x)
        q_x = pt.distributions.normal.Normal(0, 1)

    c = pt.divide(p_x(sampling_range), q_x.log_prob(sampling_range).exp()).max()
    samples = q_x.sample((num_sample,))
    samples = samples[(samples > sampling_range.min()) & (samples < sampling_range.max())]

    mask = pt.divide(p_x(samples), c*q_x.log_prob(samples).exp())
    u = pt.distributions.uniform.Uniform(0, 1).sample(mask.shape)
    mask = pt.where(u <= mask, True, False)

    first_moment = pt.masked_select(samples, mask).mean()
    second_moment = pt.mean(pt.masked_select(samples, mask)**2)
    return first_moment, second_moment


def importance_sampling(p_x, q_x=None, sampling_range=None, num_sample=100, interval=50):
    """Perform rejection sampling from q(x) to p(x)
    Args:
        p_x: the target distribution (un-normalized)
        q_x: the proposal distribution (known pdf of torch distribution object)
        sampling_range: the range of the sampling
        num_sample: number of samples
        interval: the interval between 0 to 10 for finding the C if range is None
    Returns:
        expectation E(x) wrt. q(x)
    """
    # step 1. find constant c to make q(x) to cover target distribution by sampling
    if sampling_range is None:
        sampling_range = pt.linspace(0, 10, interval)

    if q_x is None:  # if q(x) not given, then use normal(0,1) as proposal q(x)
        q_x = pt.distributions.normal.Normal(0, 1)

    c = pt.divide(p_x(sampling_range), q_x.log_prob(sampling_range).exp()).max()
    samples = q_x.sample((num_sample,))
    samples = samples[(samples > sampling_range.min()) & (samples < sampling_range.max())]

    weights = pt.divide(p_x(samples), c*q_x.log_prob(samples).exp())

    first_moment = pt.sum(samples * weights) / pt.sum(weights)
    second_moment = pt.sum(samples**2 * weights) / pt.sum(weights)
    return first_moment, second_moment


def importance_sampling_old(f_x, q_x=None, sampling_range=None, num_sample=100, interval=50):
    """Perform importance sampling from q(x) to p(x)
    Args:
        p_x: the target distribution (un-normalized)
        q_x: the proposal distribution (known pdf of torch distribution object)
        sampling_range: the range of the sampling
        num_sample: number of samples
        interval: the interval between 0 to 10 for finding the C if range is None
    Returns:
        expectation E(x) wrt. q(x)
    """
    # step 1. find constant c to make q(x) to cover target distribution by sampling
    if sampling_range is None:
        sampling_range = pt.linspace(0, 10, interval)

    if q_x is None:  # if q(x) not given, then use normal(0,1) as proposal q(x)
        q_x = pt.distributions.normal.Normal(0, 1)

    #p_x = lambda x: pt.exp(f_x(x))
    samples = q_x.sample((num_sample,))
    samples = samples[(samples > sampling_range.min()) & (samples < sampling_range.max())]

    z = pt.sum(f_x(samples))
    weights = pt.divide(f_x(samples), q_x.log_prob(samples).exp())
    first_moment = pt.sum(f_x(samples) * weights) / z
    second_moment = pt.sum(f_x(samples)**2 * weights) / z

    return first_moment, second_moment


if __name__ == '__main__':
    # test AIS
    f_0 = lambda x: pt.exp(-(x-3)**2 / (2 * 2**2))  # target distribution (unnormalized)
    f_n = lambda x: pt.exp(-(x-1)**2 / (2 * 2**2))  # proposal distribution (unnormalized)
    q_n = pt.distributions.normal.Normal(2, 2)  # the proposal distribution (pdf)
    # mu_tol = []
    # var_tol = []
    # for i in range(50):
    #     f_m, s_m = annealed_importance_sampling(f_0, f_n, q_n, num_sample=100, interval=10)
    #     mu_tol.append(f_m)
    #     var_tol.append(s_m)
    # print(mu_tol, var_tol)

    # Test importance sampling accuracy
    f_x = lambda x: 3*x + 2

    # Test rejection sampling accuracy
    e = []
    for i in range(10):
        # res = rejection_sampling(f_0, q_x=pt.distributions.normal.Normal(i, 3),
        #                          sampling_range=pt.linspace(0, 10, 100),
        #                          num_sample=1000)
        res = importance_sampling(f_0, q_x=q_n, sampling_range=pt.linspace(0, 6, 100), num_sample=1000)
        e.append(res)
    print(e)
