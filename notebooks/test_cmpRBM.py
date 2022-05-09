import os  # to handle path information
# import sys
# sys.path.append(os.path.abspath('..'))
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp,log,sqrt
import torch as pt
import arrangements as ar
import emissions as em
import full_model as fm
import spatial as sp
import copy
import evaluation as ev
import pandas as pd
import seaborn as sb
import time



def make_cmpRBM(width=10,K=5,N=200,
            theta_mu=20,theta_w=2,sigma2=0.5):
    """ Generates a smooth convolutional mpRBM
    Args:
        width (int, optional): [description]. Defaults to 10.
        K (int, optional): [description]. Defaults to 5.
        theta_mu (int, optional): [description]. Defaults to 20.
        theta_w (int, optional): [description]. Defaults to 2.
        sigma2 (float, optional): [description]. Defaults to 0.5.
        do_plot (bool): Make a plot of the first 10 samples?
    """
    # Step 1: Create the true model
    grid = sp.SpatialGrid(width=width,height=width)
    W = grid.get_edge_connectivity()


    arrangeT = ar.cmpRBM_pCD(W, K=K)
    emissionT = em.MixVMF(K=K, N=N, P=grid.P)

    # Step 2: Initialize the parameters of the true model
    arrangeT.random_smooth_pi(grid.Dist,theta_mu=theta_mu)
    arrangeT.theta_w = pt.tensor(theta_w)
    emissionT.random_params()
    emissionT.sigma2=pt.tensor(sigma2)
    MT = fm.FullModel(arrangeT,emissionT)

    # Step 3: Plot the prior of the true mode
    # plt.figure(figsize=(7,4))
    # grid.plot_maps(exp(arrangeT.logpi),cmap='jet',vmax=1,grid=[2,3])
    # cluster = np.argmax(arrangeT.logpi,axis=0)
    # grid.plot_maps(cluster,cmap='tab10',vmax=9,grid=[2,3],offset=6)

    # Step 4: Generate data by sampling from the above model
    U = MT.arrange.sample(num_subj=N,burnin=19) # These are the subjects
    Ytrain = MT.emission.sample(U.numpy()) # This is the training data
    Ytest = MT.emission.sample(U.numpy())  # Testing data

    # Plot first 10 samples
    if do_plot:
        plt.figure(figsize=(10,4))
        grid.plot_maps(U[0:10],cmap='tab10',vmax=K,grid=[2,5])

    return Ytrain, Ytest, U, MT


if __name__ == '__main__':
    # compare_gibbs()
    # train_rbm_to_mrf2('notebooks/sim_500.pt',n_hidden=[30,100],batch_size=20,n_epoch=20,sigma2=0.5)
    test_epos_meanfield()
    # test_sample_multinomial()
    # train_RBM()
