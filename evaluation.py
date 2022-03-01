"""
Evaluation - implements evaluation of emission, arrangement, or full models 
Assumes that data, likelihoods, and estimates comes as NxKxP tensors
First are basic functions for save evaluation - 
Second are more complex functions that use different criteria 
""" 
import torch as pt 
import numpy as np
from full_model import FullModel
from emissions import MixGaussianExp, MixGaussian, MixGaussianGamma, MixVMF
from notebooks.test_emissions import generate_data


def u_abserr(U,uhat):
    """Absolute error on U 
    Args:
        U (tensor): Real U's 
        uhat (tensor): Estimated U's from arangement model 
    """
    return pt.mean(pt.abs(U-uhat))


def u_prederr(U, uhat):
    """Prediction error on U
    Args:
        U: The true U (tensor like)
        uhat: The predicted U's from emission model
    Returns:
        the averaged prediction error
    """
    return pt.count_nonzero(pt.abs(U-uhat))/U.numel()


def coserr(U_pred, data, prediction):
    """Compute the cosine distance between the data to the predicted V's
    Args:
        U_pred: the predicted U's from the trained emission model
        data: the training data, with a shape (num_sub, N, P)
        prediction: the predicted V's
    Returns:
        the cosine distance. 0 indicates the same direction;
        1 means opposite direction. the lower the value the better
    """
    # standardise V to unit length
    prediction = prediction / pt.sqrt(pt.sum(prediction ** 2, dim=0))
    # standardise data to unit norm
    data = data / pt.sqrt(pt.sum(data**2, dim=1)).unsqueeze(1).repeat(1, data.shape[1], 1)
    cos_distance = 1 - pt.matmul(data.transpose(1, 2), prediction)
    cos_distance = pt.gather(cos_distance, dim=2, index=U_pred.unsqueeze(dim=2))

    return pt.mean(cos_distance)


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


def logpY(emloglik,Uhat,offset='P'):
    """Averaged log of p(Y|U)
    For save computation (to prevent underflow or overflow) 
    p(y|u) either gets a constant
    """
    if offset is None:
        pyu = pt.exp(emloglik)
    elif offset=='P':
        pyu = pt.softmax(emloglik,dim=1)
    elif type(offset) is float:
        pyu = pt.exp(emloglik-offset)
    else:
        raise(NameError('offset needs to be P,None, or floatingpoint'))
    py = pt.sum(pyu * Uhat,dim=1)
    return pt.mean(pt.log(py),dim=(0,1))


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


def evaluate_full_arr(data,Uhat,crit='logpY',offset='P'):
    """Evaluates an arrangement model new data set using pattern completion from partition to partition, using a leave-one-partition out crossvalidation approach.
    Args:
        data (tensor): Emission log-liklihood, Us (depends on crit)
        Uhat (tensor): Probility for each node (expected U) 
        crit (str): 'logpy','u_abserr'
    Returns:
        evaluation citerion: [description]
    """
    if type(data) is np.ndarray:
        data=pt.tensor(data,dtype=pt.get_default_dtype())
    if crit=='logpY': 
        U = pt.softmax(data,dim=1)
        emloglik = data
        return logpY(emloglik,Uhat)
    elif crit == 'u_abserr': 
        U = data
        return u_abserr(U,Uhat)


def evaluate_completion_arr(arM,data,part,crit='logpY',offset='P'):
    """Evaluates an arrangement model new data set using pattern completion from partition to partition, using a leave-one-partition out crossvalidation approach.
    Args:
        arM (ArrangementModel): [description]
        data (tensor): Emission log-liklihood or U-estimates (depends on crit)
        part (Partitions): P tensor with partition indices 
        crit (str): 'logpy','u_abserr'
    Returns:
        evaluation citerion: [description]
    """
    if type(data) is np.ndarray:
        data=pt.tensor(data,dtype=pt.get_default_dtype())
    if crit=='logpY': 
        U = pt.softmax(data,dim=1)
        emloglik = data
        if offset is None:
            pyu = pt.exp(emloglik)
        elif offset=='P':
            pyu = pt.softmax(emloglik,dim=1)
        elif type(offset) is float:
            pyu = pt.exp(emloglik-offset)
        else:
            raise(NameError('offset needs to be P,None, or floatingpoint'))
    elif crit == 'u_abserr': 
        U = data
    else:
        raise(NameError('unknown criterion'))
    N = U.shape[0]
    num_part = part.max()+1
    loss = pt.zeros(arM.P)
    for k in range(num_part):
        ind = part==k
        U0 = pt.clone(U)
        U0[:,:,ind] = 1./arM.K # Agnostic input
        Uhat,_ = arM.Estep(U0,gather_ss=False) 
        if crit=='abserr':
            loss[ind] = pt.mean(pt.abs(U[:,:,ind] - Uhat[:,:,ind]),dim=(0,1))
        elif crit=='logpY':
            py = pt.sum(pyu[:,:,ind] * Uhat[:,:,ind],dim=1)
            loss[ind] = pt.mean(pt.log(py),dim=0)
    return pt.mean(loss) # average across vertices 


def evaluate_completion_emission(emissionM, data, U_true, U_predict=None, crit='u_prederr'):
    """ Evaluates an emission model on a given data set using a given
        criterion. This data set can be the training dataset (includes
        U and signal if applied), or a new dataset
        given criterion
    Args:
        emissionM: this is actually the full model with freezing arrangement model
        data: The data used to evaluate, shape (num_subj, N, P)
        U_true: the true U's
        crit: the criterion to be used to evaluate the models
    Returns:
        evaluation results
    """
    if type(data) is np.ndarray:
        data = pt.tensor(data, dtype=pt.get_default_dtype())
    if U_predict is None:
        U_predict, _ = emissionM.Estep(Y=data)

    # Switching between the evaluation criterion
    if crit == 'u_prederr':
        U_predict = pt.argmax(U_predict, dim=1)
        perm = permute(np.unique(U_predict))
        min_err = 1
        for idx in perm:
            this_U_pred = np.choose(U_predict, idx)
            u_abserr = u_prederr(U_true, this_U_pred)
            if u_abserr < min_err:
                min_err = u_abserr
        eval_res = min_err
    elif crit == 'coserr':
        # The criterion to compute the cosine angle between data
        # and the predicted Vs. \sum (Y.T @ Y_pred)
        U_predict = pt.argmax(U_predict, dim=1)
        perm = permute(np.unique(U_predict))
        V_pred = emissionM.emission.V

        # # Mean centering
        # V_pred = V_pred - pt.mean(V_pred, dim=0)  # Mean centering
        # data = data - pt.mean(data, dim=1).unsqueeze(1).repeat(1, data.shape[1], 1)
        min_err = 2
        for idx in perm:
            this_U_pred = np.choose(U_predict, idx)
            cos_err = coserr(this_U_pred, data, V_pred)
            if cos_err < min_err:
                min_err = cos_err
        eval_res = min_err
    elif crit == 'adjustSSE':
        # TODO: calculate the adjusted mean squared error
        V_pred = emissionM.emission.V
        # V_pred = V_pred - pt.mean(V_pred, dim=0)  # Mean centering
        if emissionM.emission.name == 'VMF':
            sse = mean_adjusted_sse(data, V_pred, U_predict, adjusted=True, soft_assign=False)
        else:
            sse = mean_adjusted_sse(data, V_pred, U_predict, adjusted=False, soft_assign=False)
        eval_res = sse
    else:
        raise NameError('The given criterion must be specified!')

    return eval_res


# if __name__ == '__main__':
#     # Evaluate emission models
#     num_sub = 10
#     P = 100
#     K = 5
#     N = 20
#
#     # Step 1. generate the training dataset from VMF model given a signal length
#     signal = pt.distributions.exponential.Exponential(0.5).sample((num_sub, P))
#     Y_train, Y_test, signal_true, U, MT = generate_data(0, k=K, dim=N, p=P, signal_strength=signal, do_plot=True)
#     # standardise Y to unit length for VMF
#     Y_train_vmf = Y_train / pt.sqrt(pt.sum(Y_train ** 2, dim=1)).unsqueeze(1).repeat(1, Y_train.shape[1], 1)
#     Y_test_vmf = Y_test / pt.sqrt(pt.sum(Y_test ** 2, dim=1)).unsqueeze(1).repeat(1, Y_test.shape[1], 1)
#
#     # Step 2a. Fit the competing emission model using the training data
#     emissionM1 = MixGaussian(K=K, N=N, P=P)
#     emissionM2 = MixGaussianExp(K=K, N=N, P=P)
#     emissionM3 = MixVMF(K=K, N=N, P=P, uniform_kappa=False)
#     M1 = FullModel(MT.arrange, emissionM1)
#     M2 = FullModel(MT.arrange, emissionM2)
#     M3 = FullModel(MT.arrange, emissionM3)
#     M1, _, _, Uhat1_train = M1.fit_em(Y=Y_train, iter=100, tol=0.00001, fit_arrangement=False)
#     M2, _, _, Uhat2_train = M2.fit_em(Y=Y_train, iter=100, tol=0.00001, fit_arrangement=False)
#     M3, _, _, Uhat3_train = M3.fit_em(Y=Y_train_vmf, iter=100, tol=0.00001, fit_arrangement=False)
#
#     # Step 2b. Predict test data using the trained model
#     Uhat1_test, _ = M1.Estep(Y=Y_test)
#     Uhat2_test, _ = M2.Estep(Y=Y_test, signal=signal_true)
#     Uhat3_test, _ = M3.Estep(Y=Y_test_vmf)
#
#     # import plotly.graph_objects as go
#     # from plotly.subplots import make_subplots
#     #
#     # fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}],
#     #            [{'type': 'surface'}, {'type': 'surface'}]], subplot_titles=["True", "GMM", "GME", "VMF"])
#     #
#     # fig.add_trace(go.Scatter3d(x=Y_train[0, 0, :], y=Y_train[0, 1, :], z=Y_train[0, 2, :],
#     #                            mode='markers', marker=dict(size=3, opacity=0.7, color=U[0])), row=1, col=1)
#     # fig.add_trace(go.Scatter3d(x=Y_train[0, 0, :], y=Y_train[0, 1, :], z=Y_train[0, 2, :],
#     #                            mode='markers', marker=dict(size=3, opacity=0.7, color=pt.argmax(Uhat1_train, dim=1)[0])), row=1, col=2)
#     # fig.add_trace(go.Scatter3d(x=Y_train[0, 0, :], y=Y_train[0, 1, :], z=Y_train[0, 2, :],
#     #                            mode='markers', marker=dict(size=3, opacity=0.7, color=pt.argmax(Uhat2_train, dim=1)[0])), row=2, col=1)
#     # fig.add_trace(go.Scatter3d(x=Y_train[0, 0, :], y=Y_train[0, 1, :], z=Y_train[0, 2, :],
#     #                            mode='markers', marker=dict(size=3, opacity=0.7, color=pt.argmax(Uhat3_train, dim=1)[0])), row=2, col=2)
#     #
#     # fig.update_layout(title_text='Comparison of fitting', height=800, width=800)
#     # fig.show()
#
#     import matplotlib.pyplot as plt
#     import seaborn as sb
#
#     fig1 = plt.figure(figsize=(10, 10))
#     plt.subplot(2, 2, 1)
#     sb.scatterplot(x=Y_train[0, 0, :], y=Y_train[0, 1, :], hue=U[0], palette="deep")
#     plt.title('True data (first two dimensions)')
#     plt.subplot(2, 2, 2)
#     sb.scatterplot(x=Y_train[0, 0, :], y=Y_train[0, 1, :], hue=pt.argmax(Uhat1_train, dim=1)[0], palette="deep")
#     plt.title('GMM')
#     plt.subplot(2, 2, 3)
#     sb.scatterplot(x=Y_train[0, 0, :], y=Y_train[0, 1, :], hue=pt.argmax(Uhat2_train, dim=1)[0], palette="deep")
#     plt.title('GME')
#     plt.subplot(2, 2, 4)
#     sb.scatterplot(x=Y_train[0, 0, :], y=Y_train[0, 1, :], hue=pt.argmax(Uhat3_train, dim=1)[0], palette="deep")
#     plt.title('VMF')
#     fig1.suptitle('Fitting results on training data', fontsize=16)
#
#     fig2 = plt.figure(figsize=(10, 10))
#     plt.subplot(2, 2, 1)
#     sb.scatterplot(x=Y_test[0, 0, :], y=Y_test[0, 1, :], hue=U[0], palette="deep")
#     plt.title('True data (first two dimensions)')
#     plt.subplot(2, 2, 2)
#     sb.scatterplot(x=Y_test[0, 0, :], y=Y_test[0, 1, :], hue=pt.argmax(Uhat1_test, dim=1)[0], palette="deep")
#     plt.title('GMM')
#     plt.subplot(2, 2, 3)
#     sb.scatterplot(x=Y_test[0, 0, :], y=Y_test[0, 1, :], hue=pt.argmax(Uhat2_test, dim=1)[0], palette="deep")
#     plt.title('GME')
#     plt.subplot(2, 2, 4)
#     sb.scatterplot(x=Y_test[0, 0, :], y=Y_test[0, 1, :], hue=pt.argmax(Uhat3_test, dim=1)[0], palette="deep")
#     plt.title('VMF')
#     fig2.suptitle('Fitting results on test data', fontsize=16)
#
#     plt.show()
#
#     # Step 3a. evaluate the fitted emission (actually the full model with
#     # freezing the arrangement model) models by a given criterion.
#     criterion = ['u_prederr', 'coserr', 'adjustSSE']
#     res = pt.empty(len(criterion), 3)
#     for c in range(len(criterion)):
#         acc1 = evaluate_completion_emission(M1, Y_test, U_true=U, crit=criterion[c])
#         acc2 = evaluate_completion_emission(M2, Y_test, U_true=U, crit=criterion[c])
#         acc3 = evaluate_completion_emission(M3, Y_test, U_true=U, U_predict=Uhat3_test, crit=criterion[c])
#         res[c, 0] = acc1
#         res[c, 1] = acc2
#         res[c, 2] = acc3
#         print(acc1, acc2, acc3)
#
#     print(res)
