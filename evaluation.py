"""
Evaluation - implements evaluation of emission, arrangement, or full models
Assumes that data, likelihoods, and estimates comes as NxKxP tensors
First are basic functions for save evaluation -
Second are more complex functions that use different criteria
"""
import torch as pt
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def nmi(U,Uhat):
    """Compute the normalized mutual information score
    Args:
        U: The real U's
        Uhat: The estimated U's from fitted model
    Returns:
        the normalized mutual information score
    """
    return 1-metrics.normalized_mutual_info_score(U, Uhat)


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
        return 1.0

    return 1 - 2.0*(n_11*n_00 - n_10*n_01)/((n_11+n_10)*(n_10+n_00)+(n_11+n_01)*(n_01+n_00))


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
        the averaged prediction error
    """
    U_true = pt.zeros(uhat.shape)
    U_true = U_true.scatter_(1, U.unsqueeze(1), 1)
    if expectation:
        return pt.mean(pt.abs(U_true - uhat)).item()
    else:
        uhat = pt.argmax(uhat, dim=1)
        return pt.count_nonzero(pt.abs(U-uhat))/U.numel()


def coserr(Y, V, U, adjusted=False, soft_assign=True):
    """Compute the cosine distance between the data to the predicted V's
    Args:
        Y: the test data, with a shape (num_sub, N, P)
        V: the predicted mean directions
        U: the predicted U's from the trained emission model (in multinomial notation)
        adjusted: Adjusted for the length of the data vector?
        soft_assign: Compute the expected mean cosine error if True; Otherwise, False
    Returns:
        the averaged expected cosine distance. 0 indicates the same direction;
        1 - orthogonal; 2 - opposite direction
    """
    # standardise V and data to unit length
    V = V / pt.sqrt(pt.sum(V ** 2, dim=0))
    Ynorm2 = pt.sum(Y**2, dim=1, keepdim=True)
    Ynorm = pt.sqrt(Ynorm2)

    if adjusted:
        # ||Y_i||-(V_k)T(Y_i)
        cos_distance = Ynorm2 - pt.matmul(V.T, Y * Ynorm)
    else:
        # 1-(V_k)T(Y_i/||Y_i||)
        cos_distance = 1 - pt.matmul(V.T, Y/Ynorm)

    if soft_assign:  # Calculate the expected cosine error
        cos_distance = pt.sum(cos_distance * U, dim=1)
    else:
        # Calculate the argmax U_hat (hard assignments)
        idx = pt.argmax(U, dim=1, keepdim=True)
        U_max = pt.zeros_like(U).scatter_(1, idx, 1.)
        cos_distance = pt.sum(cos_distance * U_max, dim=1)

    return pt.nanmean(cos_distance, dim=1)


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
    """The recursive algorithm to find all permutations using
       back-tracking algorithm
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
        soft_assign: True - expected U over all k clusters; False - if take the argmax
                     from the k probability
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
    perm = permute(np.unique(U_predict))
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
    
    # Initialize index array 
    indx = np.empty((K,),np.int)
    for i in range(K):
        ind = np.unravel_index(np.nanargmax(Corr),Corr.shape)
        indx[ind[0]]=ind[1]
        Corr[ind[0],:]=pt.nan
        Corr[:,ind[1]]=pt.nan
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
