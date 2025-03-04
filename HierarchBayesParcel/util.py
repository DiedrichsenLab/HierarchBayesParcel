#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for Utility functions for Hierarchical Bayesian Parcellation

Author: jdiedrichsen
"""
import pickle
import numpy as np
import torch as pt
import pandas as pd
import nibabel as nb 
from sklearn.cluster import KMeans

def load_group_parcellation(fname, index=None, marginal=False,
                            device=None):
    """ Loads a group parcellation prior from a list of pre-trained model

    Args:
        fname (str): File name of pre-trained model
        index (int): Index of the model to load. If None,
            loads the model with the highest log-likelihood
            by default.
        marginal (bool): If True, return marginal probability.
            Else, return the logpi.
        device (str): Device to load the model to. Current
            support 'cuda' and 'cpu'.

    Returns:
        U: the logpi of arrangement model
        info_reduced: Data Frame with information
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
    if marginal:
        U = select_model.marginal_prob()
    else:
        U = select_model.arrange.logpi
        U = select_model.arrange.map_to_full(U)

    return U, info_reduced


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


def report_cuda_memory():
    """Reports the current memory usage of the GPU
    """
    if pt.cuda.is_available():
        current_device = 'cuda:1'

        ma = pt.cuda.memory_allocated(current_device)/1024/1024
        mma = pt.cuda.max_memory_allocated(current_device)/1024/1024
        mr = pt.cuda.memory_reserved(current_device)/1024/1024
        print(f'Allocated:{ma:.2f} MB, MaxAlloc:{mma:.2f} MB, Reserved {mr:.2f} MB')

def find_maximum_divisor(N):
    """Finds the maximum divisor of a number
    Args:
        N: a integer number

    Returns:
        i: the maximum divisor of N
    """
    for i in range(N-1, 0, -1):
        if N % i == 0:
            return i

def indicator(index_vector, positive=False):
    """Indicator matrix with one
    column per unique element in vector

    Args:
        index_vector (numpy.ndarray): n_row vector to
            code - discrete values (one dimensional)
        positive (bool): should the function ignore zero
            negative entries in the index_vector?
            Default: false

    Returns:
        indicator_matrix (numpy.ndarray): nrow x nconditions
            indicator matrix

    """
    c_unique = np.unique(index_vector)
    n_unique = c_unique.size
    rows = index_vector.shape[0]
    if positive:
        c_unique = c_unique[c_unique > 0]
        n_unique = c_unique.size
    indicator_matrix = np.zeros((rows, n_unique))
    for i in np.arange(n_unique):
        indicator_matrix[index_vector == c_unique[i], i] = 1
    return indicator_matrix


def make_random_parcellation(num_parcel, surf_file, mask_file):
    """Randomly make surface cortical parcellation by given number of 
       parcels using input surf.gii and mask files 

    Args:
        num_parcel (int): the desired number of parcels
        surf_file (str): the file path of the surf.gii used to 
                generate parcellation on
        mask_file (str): the file path of the mask file for 
                the surface

    Returns:
        random parcellation: mask.darrays[0].data is a 1d vector of
                the random parcellation starting with 1 to num_parcel,
                label 0 indicates the medial wall

    Usage:
    >>> surf_file = 'tpl-fs32k_hemi-L_midthickness.surf.gii'
    >>> mask_file = '/tpl-fs32k_hemi-L_mask.label.gii'
    >>> rand_par = make_random_parcellation(50, surf_file, mask_file)

    `rand_par` is a random parcellation with 50 parcels that spatially
    distributed for left hemisphere
    """
    # Load surface data / medial wall mask
    surf = nb.load(surf_file)
    mask = nb.load(mask_file)

    # Get indices of vertices in medial wall mask 
    mask_vertices = np.where(mask.darrays[0].data == 1)[0]
    ### Only calculate k-means on vertices outside the medial wall 
    vertices = surf.darrays[0].data[mask_vertices,:]
    # Set k-means params
    kmeans = KMeans(n_clusters=num_parcel, max_iter=1000, n_init=5)
    label = kmeans.fit_predict(vertices)

    # Replace medial wall mask values with new label values
    mask.darrays[0].data[mask_vertices] = label +1

    # Just return the parcellation array
    return mask.darrays[0].data
