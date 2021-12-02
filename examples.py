# Example Models
import numpy as np
from numpy import exp,sqrt,log
import matplotlib.pyplot as plt
import copy
import emissions as em
import arrangements as ar
import spatial as sp
from full_model import FullModel
# import seaborn as sns

def simulate_potts_gauss_grid():

    # Step 1: Create the true model
    grid = sp.SpatialGrid(width=10,height=10)
    arrangeT = ar.PottsModel(grid.W, K=5)
    emissionT = em.MixGaussian(K=5, N=40, P=grid.P)
    # Step 2: Initialize the parameters of the true model
    arrangeT.random_smooth_pi(grid.Dist,theta_mu=50)
    emissionT.random_params()
    emissionT.sigma2=3

    # Step 3: Plot the prior of the true mode
    plt.figure(figsize=(10,2))
    grid.plot_maps(exp(arrangeT.logpi),cmap='jet',vmax=1,grid=[2,3])
    cluster = np.argmax(arrangeT.logpi,axis=0)
    grid.plot_maps(cluster,cmap='tab10',vmax=9,grid=[2,3],offset=6)

    # Step 4: Generate data by sampling from the above model
    U = arrangeT.sample_new(num_subj=10)
    Y = emissionT.sample(U)
    grid.plot_maps(U)

    # Step 5: Generate new models for fitting
    arrangeM = ar.PottsModel(grid.W, K=5)
    emissionM = em.MixGaussian(K=5, N=40, P=grid.P)
    arrangeT.random_smooth_pi(grid.Dist,theta_mu=4)
    emissionT.random_params()

    # Step 4: Estimate the parameter thetas to fit the new model using EM
    M = FullModel(arrangeM, emissionM)
    ll, theta = M.fit_em(iter=1000, tol=0.001)
    plt.lineplot(ll, color='b')
    print(theta)


if __name__ == '__main__':
    simulate_potts_gauss_grid()
