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


def simulate_potts_gauss_duo():
    # Step 1: Create the true model
    W = np.array([[0,1],[1,0]])
    pi = np.array([[0.6,0.25],[0.1,0.25],[0.1,0.25],[0.2,0.25]])
    arrangeT = ar.PottsModel(W, K=4)
    emissionT = em.MixGaussian(K=4, N=5, P=2)
    # Step 2: Initialize the parameters of the true model
    arrangeT.logpi=log(pi)
    arrangeT.theta_w = 2
    emissionT.random_params()
    emissionT.sigma2=3

    # Step 3: Generate data by sampling from the above model
    U = arrangeT.sample(num_subj=100,burnin=6)
    # U = arrangeT.sample(num_subj=10)
    Y = emissionT.sample(U)

    # Plot the joint distribution of the hidden variables
    df = pd.DataFrame({'U1':U[:,0],'U2':U[:,1]})
    T=pd.pivot_table(df,index='U1',values='U1',columns='U2',aggfunc=len)
    print(T)

    plt.figure()
    T[np.isnan(T)]=0
    plt.imshow(T.to_numpy())

    # Step 5: Generate new models for fitting
    arrangeM = ar.PottsModel(W, K=4)
    emissionM = copy.deepcopy(emissionT)

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


if __name__ == '__main__':
    simulate_potts_gauss_duo()
