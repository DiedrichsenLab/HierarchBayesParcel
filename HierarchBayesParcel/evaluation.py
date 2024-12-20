#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/14/2021
Evaluation - implements evaluation of emission, arrangement, or full models
Assumes that data, likelihoods, and estimates comes as NxKxP tensors
First are basic functions for save evaluation -
Second are more complex functions that use different criteria

Author: dzhi, jdiedrichsen
"""
import torch as pt
import numpy as np
from sklearn import metrics
import warnings
import time

def pt_nanstd(tensor, dim=None):
    """Compute the standard deviation of tensor along the
       specified dimension.

    Args:
        tensor (torch.Tensor):
            the given pytorch tensor
        dim (int): the dimension along which to compute the
            standard deviation. If None, compute the
            standard deviation of the flattened tensor.

    Returns:
        the standard deviation of tensor along the specified
    """
    if dim is not None:
        return pt.sqrt(pt.nanmean(pt.pow(pt.abs(tensor -
            pt.nanmean(tensor, dim=dim, keepdim=True)), 2),
            dim=dim))
    else:
        return pt.sqrt(pt.nanmean(pt.pow(pt.abs(tensor -
            pt.nanmean(tensor)), 2)))


def nmi(U,Uhat):
    """Compute the normalized mutual information score

    Args:
        U: The real U's
        Uhat: The estimated U's from fitted model
    Returns:
        the normalized mutual information score
    """
    return 1-metrics.normalized_mutual_info_score(U, Uhat)


def dice_coefficient(labels1, labels2, label_matching=True, separate=False):
    """Compute the Dice coefficient between tow parcellations

    Args:
        labels1 (pt.Tensor): a 1d tensor of parcellation 1
        labels2 (pt.Tensor): a 1d tensor of parcellation 2
    Returns:
        the Dice coefficient between the two parcellations
    """
    if len(labels1) != len(labels2):
        raise ValueError("Length of input arrays must be the same.")

    L_1, L_2 = pt.unique(labels1), pt.unique(labels2)
    dice_matrix = pt.zeros((len(L_1), len(L_2)))

    # 1. Calculate the Dice coefficient matrix
    for i, label1 in enumerate(L_1):
        for j, label2 in enumerate(L_2):
            intersection = ((labels1 == label1) & (labels2 == label2)).sum().item()
            union = (labels1 == label1).sum().item() + (labels2 == label2).sum().item()
            dice = 2.0 * intersection / union if union != 0 else 0
            dice_matrix[i, j] = dice

    if not label_matching:
        assert pt.equal(L_1, L_2), "By selecting no label matching for " \
                                   "dice coefficient, the give parcellations" \
                                   " must be have exectly the same parcel labels."
        if separate:
            return pt.diagonal(dice_matrix)
        else:
            return pt.diagonal(dice_matrix).mean()

    else:
        # 2. Initialize
        matching_pairs, dice_coef = [], []
        max_dice = 0
        # Iterate until all labels from labels1 are matched
        while not (dice_matrix == -1).all():
            # find the best matching pair
            best_pair = (dice_matrix == pt.max(dice_matrix)).nonzero()[0]
            row, col = best_pair
            max_dice = dice_matrix[row, col].clone()

            # record matching pair and its dice value
            matching_pairs.append(best_pair)
            dice_coef.append(max_dice)

            # Delete the row/col of the matching labels
            dice_matrix[row] = -1
            dice_matrix[:,col] = -1

        # Calculate the average Dice coefficient
        if separate:
            return dice_coef
        else:
            res = sum(dice_coef) / len(dice_coef) if dice_coef else 0
            return res


def ARI(U, Uhat, sparse=True):
    """Compute the 1 - (adjusted rand index) between the two parcellations

    Args:
        U: The true U's
        Uhat: The estimated U's from fitted model
    Returns:
        the adjusted rand index score
    """
    # Get info from both U and Uhat
    sameReg_U = (U[:, None] == U).int()
    sameReg_Uhat = (Uhat[:, None] == Uhat).int()
    sameReg_U = sameReg_U.fill_diagonal_(0)
    sameReg_Uhat = sameReg_Uhat.fill_diagonal_(0)

    n_11 = (sameReg_U * sameReg_Uhat).sum()
    tmp = (1-sameReg_U) * (1-sameReg_Uhat)
    n_00 = tmp.fill_diagonal_(0).sum()
    tmp = sameReg_U - sameReg_Uhat
    tmp[tmp < 0] = 0
    n_10 = tmp.sum()
    tmp = sameReg_Uhat - sameReg_U
    tmp[tmp < 0] = 0
    n_01 = tmp.sum()

    # Special cases: empty data or full agreement (tn, fp), (fn, tp)
    if n_01 == 0 and n_10 == 0:
        return pt.tensor(1.0)

    return 2.0*(n_11*n_00 - n_10*n_01)/((n_11+n_10)*(n_10+n_00)+(n_11+n_01)*(n_01+n_00))


def BIC(loglik, N, d):
    """Bayesian Information Criterion

    Args:
        loglik: the log-likelihood of the model
        N: the number of examples in the training dataset
        d: the number of parameters in the model
    Returns:
        BIC statistic
    References:
        Page 235, The Elements of Statistical Learning, 2016.
        BIC = -2 * LL + log(N) * k
    """
    bic = -2 * loglik + pt.log(N) * d
    return bic


def cross_entropy(p, q):
    """Computes the cross-entropy between two multinomial
       distributions p and q. p and q are assumed to be have
       the same shape (nsub, K, P), or multiply broadcastable
       shapes. For example, (nsub, K, P) and (K, P) are valid shapes.

    Args:
        p (ndarray or tensor): the true probability distribution.
            typically it has a shape of (sub, K, P), where the dimension K
            reprensents the multinomial distribution, which must sum to 1
        q (ndarray or tensor): the predicted probability distribution.
            typically it has a shape of (sub, K, P), where the dimension K
            reprensents the multinomial distribution, which must sum to 1

    Returns:
        ce (float): The cross-entropy between p and q.
    """
    if type(p) is np.ndarray:
        p = pt.tensor(p, dtype=pt.get_default_dtype())
    if type(q) is np.ndarray:
        q = pt.tensor(q, dtype=pt.get_default_dtype())

    ce = -pt.sum(p * pt.nan_to_num(pt.log(q), neginf=0))
    return ce


def KL_divergence(p, q):
    """Computes the KL divergence between two multinomial distributions
       p and q. p and q are assumed to be have the same shape (nsub, K, P),
       or multiply broadcastable shapes. For example, (nsub, K, P) and (K, P)
       are valid shapes.

    Args:
        p (ndarray or tensor): the true probability distribution.
            typically it has a shape of (sub, K, P), where the dimension K
            reprensents the multinomial distribution, which must sum to 1
        q (ndarray or tensor): the predicted probability distribution.
            typically it has a shape of (sub, K, P), where the dimension K
            reprensents the multinomial distribution, which must sum to 1

    Returns:
        KL_div (float): The KL divergence between p and q.
    """
    if type(p) is np.ndarray:
        p = pt.tensor(p, dtype=pt.get_default_dtype())
    if type(q) is np.ndarray:
        q = pt.tensor(q, dtype=pt.get_default_dtype())

    kl = pt.sum(p * pt.nan_to_num(pt.log(p/q), neginf=0))
    return kl


def u_abserr(U,uhat):
    """Absolute error on U

    Args:
        U (tensor): Real U's
        uhat (tensor): Estimated U's from arrangement model
    """
    return pt.mean(pt.abs(U-uhat)).item()


def u_prederr(U, uhat, expectation=True):
    """Prediction error on U

    Args:
        U: The true U (tensor like)
        uhat: The predicted U's from emission model
        expectation: if True, calculate the expected error; Otherwise
                     calculate the hard assignment error between true
                     and the inference Uhat
    Returns:
        urpred: the averaged prediction error
    """
    U_true = pt.zeros(uhat.shape)
    U_true = U_true.scatter_(1, U.unsqueeze(1), 1)
    if expectation:
        return pt.mean(pt.abs(U_true - uhat)).item()
    else:
        uhat = pt.argmax(uhat, dim=1)
        return pt.count_nonzero(pt.abs(U-uhat))/U.numel()


def cosine_error(Y, V, U, adjusted=False, type='expected'):
    """Compute the cosine errors between the data to the predicted of the probabilistic model
    For mathematical details, see https://hierarchbayesparcel.readthedocs.io/en/latest/math.html

    Args:
        Y (pt.tensor): the test data, with a shape (num_sub, N, P) or (N,P) for one subject
        V (pt.tensor): the predicted mean directions (unit length) per parcel (N, K)
        U (pt.tensor): the expected U's from the trained emission model (n_subj,K,P) or (K,P) for group model
        adjusted (bool): Is the weight of each voxel adjusted by the magnitude of the data? If yes,
            the cosine error is 2(1-R^2)
        type (str):
            'hard': Do a hard assignment and use the V from the parcel with max probability
            'average': Compute the cosine error for the average prediction (across parcels)
            'expected': Compute the average cosine error across all predictions of the parcels
    Returns:
        Cosine Error (pt.tensor): (num_subj) tensor of cosine errors 0 (same direction) to 2 (opposite direction)
    """
    # standardise V to unit length - make sure not to change the original V
    Vn = V / pt.sqrt(pt.sum(V ** 2, dim=0))

    # If U and Y are 2-dimensional (1 subject), add the first dimension
    if Y.dim() == 2:
        Y = Y.unsqueeze(0)
    if U.dim() == 2:
        U = U.unsqueeze(0)

    # Get the norm of the data
    Ynorm2 = pt.sum(Y**2, dim=1, keepdim=True)
    Ynorm = pt.sqrt(Ynorm2)

    # Compute the prediction for each voxel under different schemes
    if type == 'hard':  # Winning parcel
        idx = pt.argmax(U, dim=1, keepdim=True)
        U_max = pt.zeros_like(U).scatter_(1, idx, 1.)
        Yhat = pt.matmul(Vn, U_max)
    elif type == 'average': # Average prediction (renormalized)
        Yhat = pt.matmul(Vn, U)
        Yhat = Yhat / pt.sqrt(pt.sum(Yhat ** 2, dim=1, keepdim=True))
    elif type == 'expected':   # Average prediction (not renormalized)
        Yhat = pt.matmul(Vn, U)
    else:
        raise ValueError("Unknown type of cosine error calculation")

    # Compute the cosine error, if adjusted, weight by the magnitude of the data
    if adjusted:
        # ||Y_i||-(V_k)T(Y_i)
        cos_error_vox = Ynorm2.squeeze(1) - pt.sum(Yhat * (Y * Ynorm),dim=1)
        cos_error= pt.nansum(cos_error_vox, dim=1)/Ynorm2.squeeze(1).nansum(dim=1)
    else:
        # 1-(V_k)T(Y_i/||Y_i||)
        cos_error_vox = 1 - pt.sum(Yhat * (Y / Ynorm),dim=1)
        cos_error = pt.nanmean(cos_error_vox, dim=1)
    return cos_error

def coserr(Y, V, U, adjusted=False, soft_assign=True):
    """ For backwards compatibility"""
    warnings.warn('coserr is deprecated, use cosine_error instead', DeprecationWarning)
    if soft_assign:
        return cosine_error(Y, V, U, adjusted=adjusted, type='expected')
    else:
        return cosine_error(Y, V, U, adjusted=adjusted, type='hard')

def coserr_2(Y, V, U, adjusted=False, soft_assign=True):
    """ For backwards compatibility"""
    warnings.warn('coserr_2 is deprecated, use cosine_error instead', DeprecationWarning)
    if soft_assign:
        return cosine_error(Y, V, U, adjusted=adjusted, type='expected')
    else:
        return cosine_error(Y, V, U, adjusted=adjusted, type='hard')


def homogeneity(Y, U_hat, soft_assign=False, z_transfer=False, single_return=True):
    """ Compute the global homogeneity measure for a given parcellation.
        The homogeneity is defined as the averaged correlation (Pearson's)
        between all vertex pairs within a parcel. Then the global homogeneity
        is calculated as the mean across all parcels.

    Args:
        Y: The underlying data to compute correlation must has a shope
            of (N, P) where N is the number of task activations and P is
            the number of brain locations.
        U_hat: the given probabilistic parcellation, shape (K, P)
        soft_assign: if True, compute the expected homogeneity. Otherwise,
        calcualte the hard assignment homogeneity.
        z_transfer: if True, apply r-to-z transformation
        single_return: if True, return the global homogeneity measure only.
            Otherwise, return the homogeneity measure per parcel, and the
            number of vertices within each parcel.

    Returns:
        g_homogeneity: the global resting-state homogeneity measure
        global_homo: the mean honogeneity for each parcel
        N: the valid (non-NaN) number of vertices in each parcel

    Notes:
        In case of homogeneity measure, a higher value indicates the
        the parcellation performs better. In case of inhomogeneity
        measure, a lower value = better parcellation performance.
    """
    # convert data to tensor
    if type(Y) is np.ndarray:
        Y = pt.tensor(Y, dtype=pt.get_default_dtype())
    if type(U_hat) is np.ndarray:
        U_hat = pt.tensor(U_hat, dtype=pt.get_default_dtype())

    # Setup - remove missing voxels and mean-centering
    idx_data = pt.all(pt.where(Y.isnan(), False, True),dim=0)
    idx_par = pt.where(U_hat.isnan(), False, True)
    idx = pt.logical_and(idx_data, idx_par)
    Y, U_hat = Y[:, idx], U_hat[idx]
    Y = Y - pt.nanmean(Y, dim=0, keepdim=True)
    P = Y.shape[1]

    # Compute the correlation matrix
    r = pt.corrcoef(Y.T)
    r.fill_diagonal_(pt.nan)

    if z_transfer:
        r = 0.5 * pt.log((1 + r) / (1 - r))
        # Convert inf to nan due to fisher transformation
        pt.nan_to_num_(r, nan=pt.nan, posinf=pt.nan, neginf=pt.nan)

    global_homo, N = [], []
    if soft_assign:  # Calculate the expected homogeneity
        # TODO: Not very sure if we need a soft version
        pass
    else:
        # Calculate the argmax U_hat (hard assignments)
        for parcel in pt.unique(U_hat):
            # Find the vertex in current parcel
            in_vertex = pt.where(U_hat == parcel)[0]
            this_r = r[in_vertex, :][:, in_vertex]
            # remove vertex with no data in current parcel
            n_k = in_vertex.numel()
            # Compute the average homogeneity within current parcel
            this_homo = this_r.nansum() / (n_k*(n_k-1))
            # Check if there is less than two vertices in the parcel
            this_homo = pt.tensor(1.0) if n_k < 2 else this_homo
            global_homo.append(this_homo)
            N.append(pt.tensor(n_k))

        global_homo = pt.stack(global_homo)  # (K, num_sub)
        N = pt.stack(N)
        assert pt.all(N >= 0)

    if single_return:
        # retrun weighted average
        return pt.nansum(global_homo * N)/N.sum()
    else:
        return global_homo, N


def task_inhomogeneity(Y, U_hat, z_transfer=True, single_return=True):
    """ Compute the global inhomogeneity measure for a given parcellation.
        The task inhomogeneity is defined as the standard deviation of
        activation z-values within each parcel. Then the task
        inhomogeneity of a contrast is calculated as the weighted mean
        across all parcels. Finally, the global task inhomogeneity is
        averaged across all task contrast.

    Args:
        Y: The underlying task contrast with a shope of (N, P) where N
            is the number of task activations and P is the number of
            brain locations.
        U_hat: the given probabilistic parcellation, shape (K, P)
        z_transfer: if True, apply r-to-z transformation
        single_return: if True, return the global homogeneity measure only.
            Otherwise, return the homogeneity measure per parcel, and the
            number of vertices within each parcel.

    Returns:
        global_homo: the mean honogeneity for each parcel
        N: the valid (non-NaN) number of vertices in each parcel

    Notes:
        In case of inhomogeneity measure, a lower value = better
        parcellation performance.
    """
    # convert data to tensor
    if type(Y) is np.ndarray:
        Y = pt.tensor(Y, dtype=pt.get_default_dtype())
    if type(U_hat) is np.ndarray:
        U_hat = pt.tensor(U_hat, dtype=pt.get_default_dtype())

    # Setup - remove missing voxels and mean-centering
    idx_data = pt.all(pt.where(Y.isnan(), False, True),dim=0)
    idx_par = pt.where(U_hat.isnan(), False, True)
    idx = pt.logical_and(idx_data, idx_par)
    Y, U_hat = Y[:, idx], U_hat[idx]

    if z_transfer:
        # first demean along the voxel dimension
        Y = Y - pt.nanmean(Y, dim=1, keepdim=True)
        Y = Y / pt.std(Y, dim=1, keepdim=True)

    global_inhomo, N = [], []
    # Calculate the argmax U_hat (hard assignments)
    for parcel in pt.unique(U_hat):
        # Find the vertex in current parcel
        in_vertex = pt.where(U_hat == parcel)[0]
        this_Y = Y[:, in_vertex]
        n_k = in_vertex.numel()
        # Compute the average inhomogeneity within current parcel
        this_inhomo = pt.std(this_Y, dim=1)
        # Check if there is less than two vertices in the parcel
        this_inhomo = pt.zeros(this_inhomo.shape) if n_k < 2 else this_inhomo
        global_inhomo.append(this_inhomo)
        N.append(pt.tensor(n_k))

    global_inhomo = pt.stack(global_inhomo)  # (K, num_contrast)
    N = pt.stack(N).reshape(-1,1)
    assert pt.all(N >= 0)

    if single_return:
        # retrun averaged task inhomo across all contrasts
        return pt.nansum((global_inhomo * N) / N.sum(), dim=0).mean()
    else:
        # return the task inhomo per task contrast
        return pt.nansum((global_inhomo * N) / N.sum(), dim=0)


def logpY(emloglik,Uhat):
    """Averaged log of <p(Y|U)>q
    Not sure anymore that this criterion makes a lot of
    """
    pyu = pt.exp(emloglik)
    py = pt.sum(pyu * Uhat,dim=1)
    return pt.mean(pt.log(py),dim=(0,1)).item()


def rmse_YUhat(U_pred, data, prediction, soft_assign=True):
    """Compute the RMSE between true and predicted V's
    Args:
        U_pred: the inferred U hat on training set using fitted model
        data: the true data, shape (num_sub, N, P)
        prediction: the predicted V's, shape (N, K)
        soft_assign: if True, compute the expected RMSE; Otherwise, argmax
    Returns:
        the RMSE
    """
    # standardise V and data to unit length if the V is not yet standardised to unit length
    prediction = prediction / pt.sqrt(pt.sum(prediction ** 2, dim=0))
    data_norm = pt.sqrt(pt.sum(data ** 2, dim=1)).unsqueeze(1)

    # ||Y_i - V_k * |Y_i|_norm ||^2
    dist = data.unsqueeze(2) - prediction.unsqueeze(2) * data_norm.unsqueeze(1)  # (subjs, N, K, P)
    dist = pt.sum(dist**2, dim=1)  # (subjs, K, P)

    if soft_assign:
        # Calculate the expected squared error
        squared_error = pt.sum(dist * U_pred, dim=1)
    else:
        # Calculate the argmax U_hat (hard assignments)
        idx = pt.argmax(U_pred, dim=1).unsqueeze(1)
        U_pred = pt.zeros_like(U_pred).scatter_(1, idx, 1.)
        squared_error = pt.sum(dist * U_pred, dim=1)

    return pt.sqrt(pt.nanmean(squared_error))


def permutations(res, nums, l, h):
    """The recursive algorithm to find all permutations using back-tracking algorithm

    Args:
        res: resultant combinations
        nums: the original array to find permutations
        l: left pointer
        h: right pointer
    Returns:
        recursive return of `res`
    """
    # Base case: add the vector to result and return
    if (l == h):
        res.append(nums.copy())
        return

    # Main recursion happens here. Permutations made
    for i in range(l, h + 1):
        # Swapping
        temp = nums[l]
        nums[l] = nums[i]
        nums[i] = temp

        # Calling permutations for next greater value of l
        permutations(res, nums, l + 1, h)

        # Backtracking
        temp = nums[l]
        nums[l] = nums[i]
        nums[i] = temp


def permute(nums):
    """Function to get the permutations

    Args:
        nums: The input array to find all permutations
    Returns:
        All permutations without replicates
    """
    # Declaring result variable
    x = len(nums) - 1
    res = []

    # Calling permutations for the first time by passing l
    # as 0 and h = nums.size()-1
    permutations(res, nums, 0, x)
    return res



def mean_adjusted_sse(data, prediction, U_hat, adjusted=True, soft_assign=True):
    """Calculate the adjusted squared error for goodness of model fitting

    Args:
        data: the real mean-centered data, shape (n_subject, n_conditions, n_locations)
        prediction: the predicted mu with shape (n_conditions, n_clusters)
        U_hat: the probability of brain location i belongs to cluster k
        adjusted: True - if calculate adjusted SSE; Otherwise, normal SSE
        soft_assign: True - expected U over all k clusters; False - if take the argmax from the k probability
    Returns:
        The adjusted SSE
    """
    # # Step 1: mean-centering the real data and the predicted mu
    # data = pt.tensor(np.apply_along_axis(lambda x: x - np.mean(x), 1, data),
    #                  dtype=pt.get_default_dtype())
    # prediction = pt.tensor(np.apply_along_axis(lambda x: x - np.mean(x), 1, prediction),
    #                        dtype=pt.get_default_dtype())

    # Step 2: get axis information from raw data
    n_sub, N, P = data.shape
    K = prediction.shape[1]
    sse = pt.empty((n_sub, K, P))  # shape [nSubject, K, P]

    # Step 3: if soft_assign is True, which means we will calculate the complete
    # expected SSE for each brain location; Otherwise, we calculate the error only to
    # the prediction that has the maximum probability argmax(p(u_i = k))
    if not soft_assign:
        out = pt.zeros(U_hat.shape)
        idx = U_hat.argmax(axis=1)
        out[np.arange(U_hat.shape[0])[:, None], idx, np.arange(U_hat.shape[2])] = 1
        U_hat = out

    # Step 4: if adjusted is True, we calculate the adjusted SSE; Otherwise normal SSE
    if adjusted:
        mag = pt.sqrt(pt.sum(data**2, dim=1))
        mag = mag.unsqueeze(1).repeat(1, K, 1)
        # standardise data to unit norm
        data = data / pt.sqrt(pt.sum(data ** 2, dim=1)).unsqueeze(1).repeat(1, data.shape[1], 1)
    else:
        mag = pt.ones(sse.shape)

    # Do real SSE calculation SSE = \sum_i\sum_k p(u_i=k)(y_real - y_predicted)^2
    YY = data**2
    uVVu = pt.sum(prediction**2, dim=0)
    for i in range(n_sub):
        YV = pt.mm(data[i, :, :].T, prediction)
        sse[i, :, :] = pt.sum(YY[i, :, :], dim=0) - 2*YV.T + uVVu.reshape((K, 1))
        sse[i, :, :] = sse[i, :, :] * mag[i, :, :]

    return pt.sum(U_hat * sse)/(n_sub * P)


def evaluate_full_arr(emM,data,Uhat,crit='cos_err'):
    """Evaluates an arrangement model new data set using pattern completion from partition to
       partition, using a leave-one-partition out crossvalidation approach.

    Args:
        emM (EmissionModel)
        data (tensor): Y-data or U true (depends on crit)
        Uhat (tensor): Probility for each node (expected U)
        crit (str): 'logpy','u_abserr', 'cos_err','Ecos_err'
    Returns:
        evaluation citerion: [description]
    """
    if type(data) is np.ndarray:
        data=pt.tensor(data,dtype=pt.get_default_dtype())
    if crit=='logpY':
        emloglik = emM.Estep(data)
        critM = logpY(emloglik,Uhat)
    elif crit == 'u_abserr':
        U = data
        critM = u_abserr(U,Uhat)
    elif crit == 'cos_err':
        critM = coserr(data, emM.V, Uhat, adjusted=False, soft_assign=False)
    elif crit == 'Ecos_err':
        critM = coserr(data, emM.V, Uhat, adjusted=False, soft_assign=True)
    return critM.mean().item()


def evaluate_completion_arr(arM,emM,data,part,crit='Ecos_err',Utrue=None):
    """Evaluates an arrangement model new data set using pattern completion from partition to
       partition, using a leave-one-partition out crossvalidation approach.

    Args:
        arM (ArrangementModel): arrangement model
        emloglik (tensor): Emission log-liklihood
        part (Partitions): P tensor with partition indices
        crit (str): 'logpY','u_abserr','cos_err'
        Utrue (tensor): For u_abserr you need to provide the true U's
    Returns:
        mean evaluation citerion
    """
    emloglik=emM.Estep(data)
    num_subj = emloglik.shape[0]
    num_part = part.max()+1
    critM = pt.zeros(arM.P)
    for k in range(num_part):
        ind = part==k
        emll = pt.clone(emloglik)
        emll[:,:,ind] = 0 # Agnostic input
        Uhat,_ = arM.Estep(emll,gather_ss=False)
        if crit=='u_abserr':
            critM[ind] = pt.mean(pt.abs(Utrue[:,:,ind] - Uhat[:,:,ind]),dim=(0,1))
        elif crit=='logpY':
            pyu = pt.exp(emloglik)
            py = pt.sum(pyu[:,:,ind] * Uhat[:,:,ind],dim=1)
            critM[ind] = pt.mean(pt.log(py),dim=0)
        elif crit=='cos_err':
            critM[ind] = coserr(data[:,:,ind], emM.V, Uhat[:,:,ind],
                        adjusted=False, soft_assign=False).mean(dim=0)
        elif crit=='Ecos_err':
            critM[ind] = coserr(data[:,:,ind], emM.V, Uhat[:,:,ind],
                        adjusted=False, soft_assign=True).mean(dim=0)
    return pt.mean(critM).item()  # average across vertices


def evaluate_U(U_true, U_predict, crit='u_prederr'):
    """ Evaluates an emission model on a given data set using a given
        criterion. This data set can be the training dataset (includes
        U and signal if applied), or a new dataset
        given criterion

    Args:
        U_predict: the predicted arrangement
        U_true: the reference (true) arrangement
        crit: the criterion to be used to evaluate the models
    Returns:
        evaluation results
    """

    # Switching between the evaluation criterion
    if crit == 'u_prederr':
        # U_predict = pt.argmax(U_predict, dim=1)
        perm = permute(np.unique(U_true))
        min_err = 1
        for idx in perm:
            this_U_true = np.choose(U_true, idx)
            u_abserr = u_prederr(this_U_true, U_predict)
            if u_abserr < min_err:
                min_err = u_abserr
        eval_res = min_err

    elif crit == 'nmi':
        U_predict = pt.argmax(U_predict, dim=1)
        eval_res = pt.zeros(U_true.shape[0])
        for i in range(U_true.shape[0]):
            eval_res[i] = nmi(U_true[i], U_predict[i])
        eval_res = pt.mean(eval_res)

    elif crit == 'ari':
        U_predict = pt.argmax(U_predict, dim=1)
        eval_res = pt.zeros(U_true.shape[0])
        for i in range(U_true.shape[0]):
            eval_res[i] = ARI(U_true[i], U_predict[i])
        eval_res = pt.mean(eval_res)

    else:
        raise NameError('The given criterion must be specified!')

    return eval_res.item()


def matching_U(U_true, U_predict):
    """Matching the parcel labels of U_hat with the true Us.

    Args:
        U_true: The ground truth Us
        U_predict: U_hat, the predicted Us
    Returns:
        The U_hat with aligned labels
    """
    U_predict = pt.argmax(U_predict, dim=1)
    perm = permute(np.unique(U_true))
    min_err = 1
    U_match = U_predict
    pred_err = None
    for idx in perm:
        this_U_pred = np.choose(U_predict, idx).clone().detach()
        err = pt.abs(U_true - this_U_pred).double()
        err[err != 0] = 1
        u_abserr = pt.mean(err).item()
        if u_abserr < min_err:
            min_err = u_abserr
            U_match = pt.clone(this_U_pred)
            pred_err = pt.mean(err, dim=1)

    return U_match, pred_err

def unravel_index(index, shape):
    """Equalavent function of np.unravel_index()
    that support Pytorch tensors

    Args:
        index: An integer array whose elements are indices
               into the flattened version of an array of
               dimensions shape.
        shape: the shape of the Tensor to use for unraveling
               indices.
    Returns:
        unraveled coords - Each array in the tuple has
        the same shape as the indices tensor.
    """
    out = []
    ind = pt.clone(index)
    for dim in reversed(shape):
        out.append(ind % dim)
        ind = pt.div(ind, dim, rounding_mode='floor')
    return tuple(reversed(out))

def matching_greedy(Y_target, Y_source):
    """ Matches the rows of two Y_source matrix to Y_target
    Using row-wise correlation and matching the highest pairs
    consecutively

    Args:
        Y_target: Matrix to align to
        Y_source: Matrix that is being aligned
    Returns:
        indx: New indices, so that YSource[indx,:]~=Y_target
    """
    K = Y_target.shape[0]
    # Compute the row x row correlation matrix
    Y_tar = Y_target - Y_target.mean(dim=1,keepdim=True)
    Y_sou = Y_source - Y_source.mean(dim=1,keepdim=True)
    Cov = pt.matmul(Y_tar,Y_sou.t())
    Var1 = pt.sum(Y_tar*Y_tar,dim=1)
    Var2 = pt.sum(Y_sou*Y_sou,dim=1)
    Corr = Cov / pt.sqrt(pt.outer(Var1,Var2))
    # convert nan to -inf if any
    pt.nan_to_num_(Corr, -pt.inf)

    # Initialize index array
    indx = pt.empty((K,), dtype=pt.long)
    for i in range(K):
        ind = unravel_index(pt.argmax(Corr), Corr.shape)
        # ind = pt.tensor(np.unravel_index(np.nanargmax(Corr.cpu().numpy()),Corr.cpu().numpy().shape))
        indx[ind[0]]=ind[1]
        Corr[ind[0],:]=-pt.inf
        Corr[:,ind[1]]=-pt.inf

    return indx

def calc_consistency(params,dim_rem = None):
    """Calculates consistency across a number of different solutions
    to a problem (after alignment of the rows/columns)
    Computes cosine similarities across the entire matrix of
    params

    Args:
        params (pt.tensor): n_sol x N x P array of data
        dim_rem (int): Dimension along which to remove the mean
            None: No mean removal
            0: Remove the overall mean of the matrices
            1: Remove the row mean
            2: Remove the column mean
    Returns:
        R (pt.tensory): n_sol x n_sol matrix of cosine similarites
    """
    data = params.clone()
    n_sol = data.shape[0]
    R = pt.ones((n_sol,n_sol))
    if dim_rem is not None:
        if dim_rem==0:
            data -= data.mean(dim=[1,2],keepdim=True)
        else:
            data -= data.mean(dim=dim_rem,keepdim=True)
    for i in range(n_sol):
        for j in range(i+1,n_sol):
            cov = pt.sum(data[i,:,:] * data[j,:,:])
            var1 = pt.sum(data[i,:,:] * data[i,:,:])
            var2 = pt.sum(data[j,:,:] * data[j,:,:])
            R[i,j]=cov/pt.sqrt(var1*var2)
            R[j,i]=R[i,j]
    return R

def extract_marginal_prob(models):
    """Extracts marginal probability values

    Args:
        models (list): List of FullMultiModel
    Returns:
        Marginal probability: (n_models x K x n_vox) tensor
    """
    n_models = len(models)
    K = models[0].emissions[0].K
    n_vox = models[0].emissions[0].P

    # Intialize data arrays
    Prob = pt.zeros((n_models,K,n_vox))

    for i,M in enumerate(models):
        pp = M.marginal_prob()
        if (pp.shape[0]!=K) | (pp.shape[1]!=n_vox):
            raise(NameError('Number of K and voxels need to be the same across models'))

        Prob[i,:,:] = pp
    return Prob

def extract_V(models):
    """ Extracts emission models vectors from a list of models

    Args:
        models (list): List of FullMultiModel
    Returns:
        list: of ndarrays (n_models x M x K) for each emission model
    """
    n_models = len(models)
    K = models[0].emissions[0].K
    n_vox = models[0].emission[0].P

    V = []

    for i,M in enumerate(models):
         for j,em in enumerate(M.emissions):
            if i==0:
                V.append(pt.zeros((n_models,em.M,K)))
            V[j][i,:,:]=em.V
    return V

def extract_kappa(model):
    """ Summarizes Kappas from a model
    All emission models need to be either uniform or non-uniform

    Args:
        model (FullMultiModel): Model to extract kapps from
    Returns:
        pt_tensor: (n_emission,K) matrix or (n_emission,) vector
    """
    n_emission = len(model.emissions)
    K = model.emissions[0].K
    if model.emissions[0].uniform_kappa:
        Kappa = pt.zeros((n_emission,))
    else:
        Kappa = pt.zeros((n_emission,K))
    for j,em in enumerate(model.emissions):
        Kappa[j]=em.kappa
    return Kappa

def align_models(models, in_place=True):
    """Aligns the marginal probabilities across different models
    if in_place = True, it changes arrangement and emission models
    ... Note that the models will be changed!

    Args:
        models (list): List of full models
        in_place (bool): Changes the models in place
    Returns:
        Marginal probability: (n_models x K x n_vox) tensor

    """
    n_models = len(models)
    K = models[0].emissions[0].K
    n_vox = models[0].emissions[0].P

    # Intialize data arrays
    Prob = pt.zeros((n_models,K,n_vox))

    for i,M in enumerate(models):
        pp = M.marginal_prob()
        if (pp.shape[0]!=K) | (pp.shape[1]!=n_vox):
            raise(NameError('Number of K and voxels need to be the same across models'))
        if i == 0:
            indx = np.arange(K)
        else:
            indx = matching_greedy(Prob[0,:,:],pp)
        Prob[i,:,:]=pp[indx,:]
        if in_place:
            models[i].arrange.logpi=models[i].arrange.logpi[indx,:]

            # Now switch the emission models accordingly
            for j,em in enumerate(M.emissions):
                em.V=em.V[:,indx]
                if (hasattr(em, 'uniform_kappa')) and (not em.uniform_kappa):
                    em.kappa = em.kappa[indx]
    return Prob


def calc_test_error(M,
                    tdata,
                    U_hats,
                    coserr_type='expected',
                    coserr_adjusted=True,
                    fit_emission = 'full'):
    """
    Evaluates the prediction (cosine-error) for group or individual parcellations
    on some new test data. For this evaluation, we need to obtain a V for new test data.
    The V is learned from N-1 subjects and then used to evaluate the left-out subjects for each parcellation.
    If fit_emission is:
        'full': The emission and individual Uhats are fully refit for each fold (arrangement model is fixed)
        'use_Uhats': Using the individual Uhats, the V is estimated using a single M-step
    Because the emission model is retrained for each subject (and that can take a bit of time),
    the function evaluate a whole set of different parcellations (group, noise-floor, individual) in one go.

    Args:
        M (full model): Full model including emission model for test data.
        tdata (ndarray or 3d-pt.tensor): (numsubj x N x P) array or tensor of test data
        U_hats (list): List of strings and/or tensors. Each element of the list can be:
             3d-pt.tensor: (nsubj x K x P) tensor of individual parcellations
             2d-pt.tensor: (K x P) tensor of group parcellation
            'group':   Group-parcellation from arrangement model
            'floor':   Noise-floor (E-step on left-out subject)
        fit_emission (str): 'full': fit the emission model and individual Uhats
                      'use_Uhats': Use the individual Uhats to derive V
        coserr_type (str): Type of cosine error (hard,average,expected)
        coserr_adjusted (bool): Adjusted cosine error?
    Returns:
        A num_eval x num_subj matrix of cosine errors, 1 row for each element in U_hats
    """
    num_subj = tdata.shape[0]
    subj = np.arange(num_subj)
    group_parc = M.marginal_prob()
    pred_err = np.empty((len(U_hats), num_subj))
    for s in range(num_subj):
        print(f'Subject:{s}', end=':')
        tic = time.perf_counter()
        # initialize the emssion model using all but one subject
        M.emissions[0].initialize(tdata[subj != s, :, :])
        # For fitting an emission model without the arrangement model,
        # we do not need multiple starting values
        M.initialize()
        if fit_emission == 'full':
            M, ll, theta, Uhat = M.fit_em(iter=200,
                                          tol=0.1,
                                          fit_emission=True,
                                          fit_arrangement=False,
                                          first_evidence=False)
        elif fit_emission == 'use_Uhats':
            if U_hats[0].ndim !=3:
                raise (NameError("When using use_Uhats, the first Uhat needs to be the individual parcellations (nsubj x K x P) "))
            # Do a single M-step using the individual Uhats
            M.emissions[0].Mstep(U_hats[0][subj != s, :, :])
        else:
            raise(NameError('fit_emission needs to be either full or use_Uhats'))
        X = M.emissions[0].X
        dat = pt.linalg.pinv(X) @ tdata[subj == s, :, :]
        for i, u in enumerate(U_hats):
            if u == 'group':
                U = group_parc
            elif u == 'floor':
                # U,ll = M.Estep(Y=pt.tensor(tdata[subj==s,:,:]).unsqueeze(0))
                M.emissions[0].initialize(tdata[subj == s, :, :])
                U = pt.softmax(M.emissions[0].Estep(
                    tdata[subj == s, :, :]), dim=1)
            elif u.ndim == 2:
                U = u
            elif u.ndim == 3:
                U = u[subj == s, :, :]
            else:
                raise (
                    NameError("U_hats needs to be 'group','floor',a 2-d or 3d-tensor"))
            a = cosine_error(dat, M.emissions[0].V, U,
                          adjusted=coserr_adjusted, type=coserr_type)
            pred_err[i, s] = a
        toc = time.perf_counter()
        print(f"{toc - tic:0.4f}s")
    return pred_err

