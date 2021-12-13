# Example Models
import numpy as np
from numpy import exp,sqrt,log
import matplotlib.pyplot as plt
import copy
import emissions as em
import arrangements as ar
import spatial as sp
from full_model import FullModel
import pandas as pd
# import seaborn as sns

def simulate_potts_gauss_grid():
    # Step 1: Create the true model
    grid = sp.SpatialGrid(width=10,height=10)
    arrangeT = ar.PottsModel(grid.W, K=5)
    emissionT = em.MixGaussian(K=5, N=40, P=grid.P)

    # Step 2: Initialize the parameters of the true model
    arrangeT.random_smooth_pi(grid.Dist,theta_mu=30)
    arrangeT.theta_w = 2
    emissionT.random_params()
    emissionT.sigma2=3

    # Step 3: Plot the prior of the true mode
    plt.figure(figsize=(7,4))
    grid.plot_maps(exp(arrangeT.logpi),cmap='jet',vmax=1,grid=[2,3])
    cluster = np.argmax(arrangeT.logpi,axis=0)
    grid.plot_maps(cluster,cmap='tab10',vmax=9,grid=[2,3],offset=6)

    # Step 4: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=10,burnin=19)
    # U = arrangeT.sample(num_subj=10)
    Y = emissionT.sample(U)

    # Plot sampling path for visualization purposes
    # plt.figure(figsize=(10,4))
    # grid.plot_maps(Uhist[:,0,:],cmap='tab10',vmax=9)
    # Plot all the subjects
    plt.figure(figsize=(10,4))
    grid.plot_maps(U,cmap='tab10',vmax=9,grid=[2,5])

    # Step 5: Generate new models for fitting
    arrangeM = ar.PottsModel(grid.W, K=5)
    emissionM = em.MixGaussian(K=5, N=40, P=grid.P)
    arrangeM.random_smooth_pi(grid.Dist,theta_mu=4)
    emissionM.random_params()

    # Step 4: Estimate the parameter thetas to fit the new model using EM
    M = FullModel(arrangeM, emissionM)
    M.emission.initialize(Y)
    # Get the (approximate) posterior p(U|Y)
    emloglik = M.emission.Estep()
    Uhat, ll_A = M.arrange.Estep(emloglik)
    Umax = np.argmax(Uhat,axis=1)
    plt.figure(figsize=(7,4))
    grid.plot_maps(Umax,cmap='tab10',vmax=9,grid=[2,5])

    # ll, theta = M.fit_em(Y, iter=1000, tol=0.001)
    plt.lineplot(ll, color='b')
    print(theta)


def simulate_potts_gauss_duo(theta_w=2,
                             sigma2 = 0.01,
                             num_subj = 100,
                             eneg_numchains=200,
                             epos_numchains=20,
                             niter = 60,
                             stepsize = 0.8,
                             fit_theta_w=True):
    """[summary]

    Args:
        theta_w (int, optional): [description]. Defaults to 2.
        sigma2 (float, optional): [description]. Defaults to 0.01.
        num_subj (int, optional): [description]. Defaults to 100.
        eneg_numchains (int, optional): [description]. Defaults to 200.
        epos_numchains (int, optional): [description]. Defaults to 20.
        niter (int, optional): [description]. Defaults to 60.
        stepsize (float, optional): [description]. Defaults to 0.8.
        fit_theta_w (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    # Step 1: Create the true model
    W = np.array([[0,1],[1,0]])
    pi = np.array([[0.6,0.25],[0.05,0.25],[0.15,0.25],[0.2,0.25]])
    arrangeT = ar.PottsModel(W, K=4)
    emissionT = em.MixGaussian(K=4, N=5, P=2)

    # Step 2: Initialize the parameters of the true model
    arrangeT.logpi=log(pi)-log(pi[-1,:])
    arrangeT.theta_w = theta_w
    emissionT.random_params()
    emissionT.sigma2=sigma2

    # Step 3: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=num_subj,burnin=6)
    # U = arrangeT.sample(num_subj=10)
    Y = emissionT.sample(U)

    # Plot the joint distribution of the hidden variables
    df = pd.DataFrame({'U1':U[:,0],'U2':U[:,1]})
    T=pd.pivot_table(df,index='U1',values='U1',columns='U2',aggfunc=len)
    print(T)
    print(T.sum(axis=1).T/num_subj)
    print(T.sum(axis=0).T/num_subj)

    # Step 4: Generate new models for fitting
    arrangeM = ar.PottsModel(W, K=4)
    arrangeM.theta_w =0
    arrangeM.eneg_numchains=eneg_numchains
    arrangeM.epos_numchains=epos_numchains
    emissionM = copy.deepcopy(emissionT)

    # Step 5: Get the emission log-liklihood:
    M = FullModel(arrangeM, emissionM)
    M.emission.initialize(Y)
    # Get the (approximate) posterior p(U|Y)
    emloglik = M.emission.Estep()

    # Step 6: Get baseline for the emission and arrangement likelihood by fitting a Indepenent model
    arrangeI = ar.ArrangeIndependent(K=4,P=2,spatial_specific=True)
    Uhat,ll_A_in = arrangeI.Estep(emloglik)
    arrangeI.Mstep(Uhat)
    Uhat,ll_A_in = arrangeI.Estep(emloglik)
    pass

    # With fixed emission model, fit the arrangement model
    numiter = 60
    theta = np.empty((numiter+1,M.arrange.nparams))
    ll_E = np.empty((numiter,num_subj))
    ll_A = np.empty((numiter,num_subj))
    theta[0,:]=M.arrange.get_params()
    for i in range(numiter):
        Uhat,ll_A[i,:] = M.arrange.epos_sample(emloglik)
        ll_E[i,:]=np.sum(emloglik*Uhat,axis=(1,2))
        M.arrange.eneg_sample()
        M.arrange.Mstep(stepsize,fit_theta_w)
        theta[i+1,:]=M.arrange.get_params()

    thetaT = arrangeT.get_params()
    iter = np.arange(numiter+1)

    # ll, theta = M.fit_em(Y, iter=1000, tol=0.001)
    plt.subplot(2,1,1)
    color = ['r','r','b','b','g','g','k']
    style = ['-',':','-',':','-',':','-.']
    marker = ['o','s','o','s','o','s','*']

    for i in range(theta.shape[1]):
        plt.plot(iter,theta[:,i],color[i]+style[i])
        plt.plot(numiter+5,thetaT[i],color[i]+marker[i])
    pass

    # Plot the likelihood
    plt.subplot(2,1,2)
    plt.plot(iter[:-1],np.mean(ll_E,axis=1)-ll_E[0,:].mean(),'k')
    plt.plot(iter[:-1],np.mean(ll_A,axis=1)-ll_A[0,:].mean(),'b')
    plt.plot(numiter+5,np.mean(ll_A_in)-ll_A[0,:].mean(),'bo')
    plt.legend(['emmision','arrange'])
    return theta,iter,thetaT


if __name__ == '__main__':
    simulate_potts_gauss_duo(sigma2 = 0.1)
    pass
