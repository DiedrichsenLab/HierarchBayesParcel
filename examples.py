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

def make_duo_model(K=4,theta_w =1, sigma2=0.1):
        # Step 1: Create the true model
    W = np.array([[0,1],[1,0]])
    pi = np.array([[0.6,0.25],[0.05,0.25],[0.15,0.25],[0.2,0.25]])
    arrangeT = ar.PottsModel(W, K=K)
    emissionT = em.MixGaussian(K=K, N=5, P=2)

    # Step 2: Initialize the parameters of the true model
    arrangeT.logpi=log(pi)-log(pi[-1,:])
    arrangeT.theta_w = theta_w
    emissionT.random_params()
    emissionT.sigma2=sigma2
    MT = FullModel(arrangeT,emissionT)
    return MT

def plot_duo_fit(theta,ll,theta_true=None,ll_true=None):
    plt.subplot(2,1,1)
    color = ['r','r','b','b','g','g','k','k']
    style = ['-',':','-',':','-',':','-.',':']
    marker = ['o','s','o','s','o','s','*','v']

    numiter, nparams = theta.shape
    iter = range(numiter)

    for i in range(nparams):
        plt.plot(iter,theta[:,i],color[i]+style[i])
        if theta_true is not None: 
            plt.plot(numiter+5,theta_true[i],color[i]+marker[i])
    pass
    plt.xlabel('Iteration')
    plt.ylabel('Theta')
    
    # Plot the likelihood
    plt.subplot(2,1,2)
    plt.plot(iter,ll[:,0]-ll[0,0],'k')
    plt.plot(iter,ll[:,1]-ll[0,1],'b')
    if ll_true is not None: 
        plt.plot(numiter+5,ll_true[0]-ll[0,0],'k*')
        plt.plot(numiter+5,ll_true[1]-ll[0,1],'b*')
    plt.legend(['emmision','arrange'])
    plt.xlabel('Iteration')
    plt.ylabel('Likelihood')


def simulate_potts_gauss_duo(theta_w=2,
                             sigma2 = 0.01,
                             num_subj = 100,
                             eneg_numchains=200,
                             epos_numchains=20,
                             numiter = 40,
                             stepsize = 0.8,
                             fit_theta_w=True):
    """Basic simulation of a two-node potts model
    with a fixed Mixed-Gaussian emission model

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
    # Make true model 
    MT = make_duo_model(K=4,theta_w=theta_w,sigma2=sigma2)
    theta_true = MT.get_params()

    # Step 3: Generate data by sampling from the above model
    U,Y = MT.sample(num_subj=num_subj)

    # Plot the joint distribution of the hidden variables
    df = pd.DataFrame({'U1':U[:,0],'U2':U[:,1]})
    T=pd.pivot_table(df,index='U1',values='U1',columns='U2',aggfunc=len)
    print(T)
    print(T.sum(axis=1).T/num_subj)
    print(T.sum(axis=0).T/num_subj)

    # Step 4: Generate new models for fitting
    arrangeM = ar.PottsModel(MT.arrange.W, K=4)
    arrangeM.theta_w =0
    arrangeM.fit_theta_w = fit_theta_w
    arrangeM.eneg_numchains=eneg_numchains
    arrangeM.epos_numchains=epos_numchains
    emissionM = copy.deepcopy(MT.emission)

    # Step 5: Get the emission log-liklihood:
    M = FullModel(arrangeM, emissionM)
    M.emission.initialize(Y)
    # Get the (approximate) posterior p(U|Y)
    emloglik = M.emission.Estep()

    # Step 6: Get baseline for the emission and arrangement likelihood by fitting a Indepenent model
    arrangeI = ar.ArrangeIndependent(K=4,P=2,spatial_specific=True)
    Uhat,ll_A_in = arrangeI.Estep(emloglik)
    arrangeI.Mstep()
    Uhat,ll_A_in = arrangeI.Estep(emloglik)
    pass

    # Step 7: Get he the baseline for emission and arrangement model 
    # from the true model 
    Uhat,ll_A_true = MT.arrange.epos_sample(emloglik)
    ll_E_true=np.sum(emloglik*Uhat,axis=(1,2))

    # Step 8: With fixed emission model, fit the arrangement model
    theta = np.empty((numiter,M.arrange.nparams))
    ll_E = np.empty((numiter,num_subj))
    ll_A = np.empty((numiter,num_subj))
    for i in range(numiter):
        theta[i,:]=M.arrange.get_params()
        Uhat,ll_A[i,:] = M.arrange.epos_sample(emloglik)
        ll_E[i,:]=np.sum(emloglik*Uhat,axis=(1,2))
        M.arrange.eneg_sample()
        M.arrange.Mstep(stepsize)

    thetaT = MT.arrange.get_params()
    ll = np.c_[ll_E.sum(axis=1),ll_A.sum(axis=1)]
    ll_true = np.array([ll_E_true.sum(),ll_A_true.sum()])
    plot_duo_fit(theta,ll,theta_true=thetaT,ll_true=ll_true)

    return theta,iter,thetaT


def simulate_potts_gauss_duo2(theta_w=2,
                             sigma2 = 0.01,
                             num_subj = 100,
                             eneg_numchains=200,
                             epos_numchains=20,
                             numiter = 60,
                             stepsize = 0.8,
                             fit_theta_w=True):
    """Simulation of a two-node potts model
    with a flexible Mixed-Gaussian emission model

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
    MT = make_duo_model(K=4,theta_w=theta_w,sigma2=sigma2)
    theta_true = MT.get_params()

    # Generate the data 
    U,Y = MT.sample(num_subj=num_subj)

    # Get the likelihood of the data under the MT 
    MT.emission.initialize(Y)
    emloglik = MT.emission.Estep()
    Uhat,ll_A_true = MT.arrange.epos_sample(emloglik)
    ll_E_true = np.sum(emloglik * Uhat,axis=(1,2))
    ll_true = [ll_A_true.sum(),ll_E_true.sum()] 

    # Step 4: Generate indepedent models for fitting
    arrangeI = ar.ArrangeIndependent(K=4, P=2,spatial_specific=True)
    emissionI = copy.deepcopy(MT.emission)
    emissionI.sigma2=0.5
    # Add a small perturbation to paramters 
    emissionI.V = emissionI.V + np.random.normal(0,0.1,emissionI.V.shape)
    emissionM = copy.deepcopy(emissionI)
    MI = FullModel(arrangeI, emissionI)

    # Step 5: Fit independent model to the data and plot
    plt.figure()
    tI_I = np.concatenate([np.arange(0,6),
                [MI.nparams-1]])
    tI_T = np.concatenate([np.arange(0,6),
                [MT.nparams-1]])
    MI,ll,theta = MI.fit_em(Y,iter=30,tol=0.001,seperate_ll=True)
    plot_duo_fit(theta[:,tI_I],ll,theta_true = theta_true[tI_T],ll_true=ll_true)

    # Step 6: Generate Potts model for fitting
    arrangeM = ar.PottsModel(MT.arrange.W, K=4)
    arrangeM.theta_w =0
    arrangeM.eneg_numchains=eneg_numchains
    arrangeM.epos_numchains=epos_numchains
    MP = FullModel(arrangeM, emissionM)

    # Step 7: Use stochastic gradient descent to fit the combined model 
    plt.figure()
    tI_T = np.concatenate([np.arange(0,6),
                [MT.nparams-1,6]])
    MP,ll,theta = MP.fit_sml(Y,iter=40,stepsize=0.8,seperate_ll=True)
    plot_duo_fit(theta[:,tI_T],ll,theta_true = theta_true[tI_T],ll_true=ll_true)
    pass



    return theta,iter

if __name__ == '__main__':
    simulate_potts_gauss_duo2(sigma2 = 0.1,numiter=10,theta_w=0)
    pass
