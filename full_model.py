from arrangements import ArrangeIndependent
from emissions import MixGaussian, MixGaussianExp, MixVMF, MixGaussianGamma
import os # to handle path information
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nibabel as nb
from nilearn import plotting
import sys
# sys.path.insert(0, "D:/python_workspace/")

class FullModel:
    def __init__(self, arrange, emission):
        self.arrange = arrange
        self.emission = emission

    def sample(self, num_subj=10):
        U = self.arrange.sample(num_subj)
        Y = self.emission.sample(U)
        return U, Y

    def fit_em(self, iter, tol):
        """ Do the real EM algorithm for the complete log likelihood for the
        combination of the arrangement model and emission model

        :param Y: the data to passing
        :param iter: the maximum iteration number
        :param tol: the delta

        :return: the resulting parameters theta and the log likelihood
        """
        # Initialize the tracking
        ll = []
        ll_A = []
        theta = np.zeros((iter, self.emission.nparams+self.arrange.nparams))
        for i in range(iter):
            # Get the (approximate) posterior p(U|Y)
            emloglik = self.emission.Estep()
            Uhat, this_ll_A = self.arrange.Estep(emloglik)
            # Compute the expected complete logliklihood
            this_ll = np.sum(Uhat * emloglik) + np.sum(this_ll_A)
            if (i > 1) and (np.abs(this_ll - ll[-1]) < tol):  # convergence
                break
            else:
                ll.append(this_ll)
                ll_A.append(np.sum(this_ll_A))

            # Updates the parameters
            self.emission.Mstep(Uhat)
            self.arrange.Mstep(Uhat)
            theta[i, :] = np.concatenate([self.emission.get_params(), self.arrange.get_params()])

        return np.asarray(ll), np.asarray(ll_A), theta[0:len(ll), :], Uhat


def _fit_full(Y):
    pass


def _plot_loglike(loglike, true_loglike, ll_a, color='b'):
    plt.figure()
    plt.plot(loglike, color=color, label='VMF')
    plt.plot(ll_a, color='g', label='GMM')
    plt.axhline(y=true_loglike, color='r', linestyle=':', label='True likelihood')
    plt.legend(loc=4)

def _plot_diff(theta_true, theta, K, name='V'):
    """ Plot the model parameters differences.

    Args:
        theta_true: the params from the true model
        theta: the estimated params
        color: the line color for the differences

    Returns: a matplotlib object to be plot

    """
    iter = theta.shape[0]
    diff = np.empty((iter, K))
    Y = np.split(theta_true, K)
    for i in range(iter):
        x = np.split(theta[i], K)
        for j in range(len(x)):
            dist = np.linalg.norm(x[j] - Y[j])
            diff[i, j] = dist
    plt.figure()
    plt.plot(diff)
    plt.title('the differences: true %ss, estimated %ss' % (name, name))


def _plt_single_param_diff(theta_true, theta, name=None):
    plt.figure()
    if name is not None:
        plt.title('The difference: true %s vs estimated %s' % (name, name))

    iter = theta.shape[0]
    theta_true = np.repeat(theta_true, iter)
    plt.plot(theta_true, linestyle='--', color='r')
    plt.plot(theta, color='b')


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
    # Step 1: mean-centering the real data and the predicted mu
    data = np.apply_along_axis(lambda x: x - np.mean(x), 1, data)
    prediction = np.apply_along_axis(lambda x: x - np.mean(x), 0, prediction)

    # Step 2: get axis information from raw data
    n_sub, N, P = data.shape
    K = prediction.shape[1]
    sse = np.empty((n_sub, K, P))  # shape [nSubject, K, P]

    # Step 3: if soft_assign is True, which means we will calculate the complete
    # expected SSE for each brain location; Otherwise, we calculate the error only to
    # the prediction that has the maximum probability argmax(p(u_i = k))
    if not soft_assign:
        out = np.zeros(U_hat.shape)
        idx = U_hat.argmax(axis=1)
        out[np.arange(U_hat.shape[0])[:, None], idx, np.arange(U_hat.shape[2])] = 1
        U_hat = out

    # Step 4: if adjusted is True, we calculate the adjusted SSE; Otherwise normal SSE
    if adjusted:
        mag = np.sqrt(np.sum(data[:, :, :] ** 2, axis=1))
        mag = np.repeat(mag[:, np.newaxis, :], K, axis=1)
    else:
        mag = np.ones(sse.shape)

    # Do real SSE calculation SSE = \sum_i\sum_k p(u_i=k)(y_real - y_predicted)^2
    YY = data**2
    uVVu = np.sum(prediction**2, axis=0)
    for i in range(n_sub):
        YV = np.dot(data[i, :, :].T, prediction)
        sse[i, :, :] = np.sum(YY[i, :, :], axis=0) - 2*YV.T + uVVu.reshape((K, 1))
        sse[i, :, :] = sse[i, :, :] * mag[i, :, :]

    return np.sum(U_hat * sse)/(n_sub * P)


def _simulate_full_GMM(K=5, P=100, N=40, num_sub=10, max_iter=50):
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False)
    emissionT = MixGaussian(K=K, N=N, P=P)
    # emissionT.random_params()

    # Step 2: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=num_sub)
    Y = emissionT.sample(U)

    # Step 2.1: Compute the log likelihood from the true model
    theta_true = np.concatenate([emissionT.get_params(), arrangeT.get_params()])
    emll_true = emissionT._loglikelihood(Y)
    Uhat, ll_a = arrangeT.Estep(emll_true)
    loglike_true = np.sum(Uhat * emll_true) + np.sum(ll_a)
    print(theta_true)

    # Step 3: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False)
    emissionM = MixGaussian(K=K, N=N, P=P, data=Y)

    # Step 4: Estimate the parameter thetas to fit the new model using EM
    M = FullModel(arrangeM, emissionM)
    ll, ll_A, theta, U_hat = M.fit_em(iter=max_iter, tol=0.001)
    SSE = mean_adjusted_sse(Y, M.emission.V, U_hat, adjusted=False)
    _plot_loglike(ll, loglike_true, ll_A, color='b')
    print('Done.')


def _simulate_full_GME(K=5, P=1000, N=40, num_sub=10, max_iter=100):
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False)
    emissionT = MixGaussianExp(K=K, N=N, P=P)
    # emissionT.random_params()

    # Step 2: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=num_sub)
    Y, signal = emissionT.sample(U)

    # Step 2.1: Compute the log likelihood from the true model
    theta_true = np.concatenate([emissionT.get_params(), arrangeT.get_params()])
    emll_true = emissionT._loglikelihood(Y, signal)
    Uhat, ll_a = arrangeT.Estep(emll_true)
    loglike_true = np.sum(Uhat * emll_true) + np.sum(ll_a)
    print(theta_true)

    # Step 3: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False)
    emissionM = MixGaussianExp(K=K, N=N, P=P, data=Y)
    emissionM.set_params([emissionM.V, emissionM.sigma2, emissionT.alpha, emissionM.beta])

    # Step 4: Estimate the parameter thetas to fit the new model using EM
    M = FullModel(arrangeM, emissionM)
    ll, ll_A, theta, U_hat = M.fit_em(iter=max_iter, tol=0.001)
    _plot_loglike(ll, loglike_true, ll_A, color='b')
    _plot_diff(theta_true[0:N*K], theta[:, 0:N*K], K, name='V')  # The mu changes
    _plt_single_param_diff(theta_true[-3-K], theta[:, -3-K], name='sigma2')  # Sigma square
    _plt_single_param_diff(theta_true[-1-K], theta[:, -1-K], name='beta')  # beta
    print('Done.')


def _simulate_full_VMF(K=5, P=100, N=40, num_sub=10, max_iter=50):
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False)
    emissionT = MixVMF(K=K, N=N, P=P, uniform=True)
    # emissionT.random_params()

    # Step 2: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=num_sub)
    Y = emissionT.sample(U)

    # Step 2.1: Compute the log likelihood from the true model
    theta_true = np.concatenate([emissionT.get_params(), arrangeT.get_params()])
    emll_true = emissionT._loglikelihood(Y)
    Uhat, ll_a = arrangeT.Estep(emll_true)
    loglike_true = np.sum(Uhat * emll_true) + np.sum(ll_a)
    print(theta_true)

    # Step 3: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False)
    emissionM = MixVMF(K=K, N=N, P=P, data=Y, uniform=False)
    # emissionM.set_params([emissionM.V, emissionM.kappa])

    # Step 4: Estimate the parameter thetas to fit the new model using EM
    M = FullModel(arrangeM, emissionM)
    ll, theta = M.fit_em(iter=max_iter, tol=0.00001)
    _plot_loglike(ll, loglike_true, color='b')
    print('Done.')


def _simulate_full_GMG(K=5, P=1000, N=40, num_sub=10, max_iter=100):
    # Step 1: Set the true model to some interesting value
    arrangeT = ArrangeIndependent(K=K, P=P, spatial_specific=False)
    emissionT = MixGaussianGamma(K=K, N=N, P=P)
    # emissionT.random_params()

    # Step 2: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=num_sub)
    Y, signal = emissionT.sample(U)

    # Step 2.1: Compute the log likelihood from the true model
    theta_true = np.concatenate([emissionT.get_params(), arrangeT.get_params()])
    emll_true = emissionT._loglikelihood(Y, signal)
    Uhat, ll_a = arrangeT.Estep(emll_true)
    loglike_true = np.sum(Uhat * emll_true) + np.sum(ll_a)
    print(theta_true)

    # Step 3: Generate new models for fitting
    arrangeM = ArrangeIndependent(K=K, P=P, spatial_specific=False)
    emissionM = MixGaussianGamma(K=K, N=N, P=P, data=Y)
    emissionM.set_params([emissionM.V, emissionM.sigma2, emissionT.alpha, emissionM.beta])

    # Step 4: Estimate the parameter thetas to fit the new model using EM
    M = FullModel(arrangeM, emissionM)
    ll, theta = M.fit_em(iter=max_iter, tol=0.001)
    _plot_loglike(ll, loglike_true, color='b')
    _plot_diff(theta_true[0:N*K], theta[:, 0:N*K], K, name='V')  # The mu changes
    _plt_single_param_diff(theta_true[-3-K], theta[:, -3-K], name='sigma2')  # Sigma square
    _plt_single_param_diff(theta_true[-3-K], theta[:, -2-K], name='alpha')  # Alpha
    _plt_single_param_diff(theta_true[-1-K], theta[:, -1-K], name='beta')  # beta
    print('Done.')


def generate_data(emission, k=2, dim=3, p=1000,
                  num_sub=10, beta=1, alpha=1, signal_type=0):
    model_name = ["GMM", "GMM_exp", "GMM_gamma", "VMF"]
    arrangeT = ArrangeIndependent(K=k, P=p, spatial_specific=False)
    U = arrangeT.sample(num_subj=num_sub)
    if signal_type == 0:
        signal = np.random.exponential(beta, (num_sub, p))
    elif signal_type == 1:
        signal = np.random.gamma(alpha, beta, (num_sub, p))
    else:
        raise ValueError("The value of signal strength must satisfy a distribution, 0 - exponential; 1 - gamma.")

    if emission == 0:  # GMM
        emissionT = MixGaussian(K=k, N=dim, P=p)
    elif emission == 1:  # GMM with exponential signal strength
        emissionT = MixGaussianExp(K=k, N=dim, P=p)
    elif emission == 2:  # GMM with gamma signal strength
        emissionT = MixGaussianGamma(K=k, N=dim, P=p)
    elif emission == 3:
        emissionT = MixVMF(K=k, N=dim, P=p)
    else:
        raise ValueError("The value of emission must be 0(GMM), 1(GMM_exp), 2(GMM_gamma), or 3(VMF).")

    np.testing.assert_equal(emissionT.name, model_name[emission])
    if (emissionT.name == "GMM_exp") or (emissionT.name == "GMM_gamma"):
        data = emissionT.sample(U, signal)
    elif emissionT.name == "VMF":
        data = emissionT.sample(U)
        signal = np.repeat(signal[:, np.newaxis, :], dim, axis=1)
        data = data * signal
    else:
        data = emissionT.sample(U)

    return data, U


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


if __name__ == '__main__':
    # _simulate_full_VMF(K=5, P=1000, N=40, num_sub=10, max_iter=50)
    # _simulate_full_GMM(K=5, P=1000, N=40, num_sub=10, max_iter=1)
    # _simulate_full_GME(K=5, P=1000, N=20, num_sub=10, max_iter=100)
    # _simulate_full_GMG(K=1, P=1000, N=20, num_sub=10, max_iter=1000)

    # Step 1: Generate data from VMF mixture model with a exponential signal strength
    K = 2
    num_sub = 10
    Y, U = generate_data(emission=3, k=K, dim=3, p=1000, num_sub=num_sub)
    U = U.astype(int)

    colors = plt.cm.rainbow(np.linspace(0, 1, K))
    # plt.scatter(Y[0, 0, :], Y[0, 1, :], Y[0, 2, :], c=colors[U[0, :]])
    # plt.axis([-5, 5, -5, 5])
    # plt.show()

    fig = go.Figure()
    fig.add_scatter3d(mode='markers', x=Y[0, 0, :], y=Y[0, 1, :], z=Y[0, 2, :],
                      marker=dict(size=3, opacity=0.7, color=colors[U[0, :]]))

    z = sample_spherical(1000)
    fig.add_scatter3d(mode='markers', x=z[0, :], y=z[1, :], z=z[2, :],
                      marker=dict(size=1, opacity=0.2, color='black'))
    fig.show()

