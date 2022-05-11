import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import matplotlib.pyplot as plt
import torch as pt
import arrangements as ar
import emissions as em
import full_model as fm
import spatial as sp
import evaluation as ev
from test_mpRBM import train_sml
import pandas as pd
import seaborn as sb
import copy


def make_mrf_data(width=10,K=5,N=20,num_subj=30,
            theta_mu=20,theta_w=2,sigma2=0.5,
            do_plot=True):
    """Generates (and plots Markov random field data)
    Args:
        width (int, optional): [description]. Defaults to 10.
        K (int, optional): [description]. Defaults to 5.
        N (int, optional): [description]. Defaults to 200.
        theta_mu (int, optional): [description]. Defaults to 20.
        theta_w (int, optional): [description]. Defaults to 2.
        sigma2 (float, optional): [description]. Defaults to 0.5.
        do_plot (bool): Make a plot of the first 10 samples?
    """
    # Step 1: Create the true model
    grid = sp.SpatialGrid(width=width,height=width)
    arrangeT = ar.PottsModel(grid.W, K=K)
    arrangeT.name = 'Potts'

    # Step 2a: Initialize the parameters of the true arrangement model
    arrangeT.logpi = grid.random_smooth_pi(theta_mu=theta_mu,K=K)
    arrangeT.theta_w = pt.tensor(theta_w)

    # Step 2b: Initialize the parameters of the true emission model
    emissionT = em.MixGaussian(K=K, N=N, P=grid.P)
    emissionT.random_params()
    emissionT.sigma2=pt.tensor(sigma2)
    MT = fm.FullModel(arrangeT,emissionT)

    # Step 3: Plot the prior of the true mode
    # plt.figure(figsize=(7,4))
    # grid.plot_maps(exp(arrangeT.logpi),cmap='jet',vmax=1,grid=[2,3])
    # cluster = np.argmax(arrangeT.logpi,axis=0)
    # grid.plot_maps(cluster,cmap='tab10',vmax=9,grid=[2,3],offset=6)

    # Step 4: Generate data by sampling from the above model
    U = MT.arrange.sample(num_subj=num_subj,burnin=19) # These are the subjects
    Ytrain = MT.emission.sample(U) # This is the training data
    Ytest = MT.emission.sample(U)  # Testing data

    # Plot first 10 samples
    if do_plot:
        plt.figure(figsize=(10,4))
        grid.plot_maps(U[0:10],cmap='tab10',vmax=K,grid=[2,5])

    return Ytrain, Ytest, U, MT , grid


def make_cmpRBM_data(width=10,K=5,N=10,num_subj=20,
            theta_mu=20,theta_w=1.0,sigma2=0.5,
            do_plot=1):
    """Generates (and plots Markov random field data)
    Args:
        width (int, optional): [description]. Defaults to 10.
        K (int, optional): [description]. Defaults to 5.
        N (int, optional): [description]. Defaults to 200.
        theta_mu (int, optional): [description]. Defaults to 20.
        theta_w (int, optional): [description]. Defaults to 2.
        sigma2 (float, optional): [description]. Defaults to 0.5.
        do_plot (int): 1: Plot of the first 10 samples 2: + sample path
    """
    # Step 1: Create the true model
    grid = sp.SpatialGrid(width=width,height=width)
    W = grid.get_neighbour_connectivity()
    W += pt.eye(W.shape[0])

    # Step 2: Initialize the parameters of the true model
    arrangeT = ar.cmpRBM_pCD(K,grid.P,Wc=W,theta=theta_w)
    arrangeT.name = 'cmpRDM'
    arrangeT.bu = grid.random_smooth_pi(K=K,theta_mu=theta_mu)

    emissionT = em.MixGaussian(K=K, N=N, P=grid.P)
    emissionT.random_params()
    emissionT.sigma2=pt.tensor(sigma2)
    MT = fm.FullModel(arrangeT,emissionT)

    # grid.plot_maps(pt.softmax(arrangeT.bu,0),cmap='hot',vmax=1,grid=[1,5])

    # Step 3: Plot the prior of the true mode
    # plt.figure(figsize=(7,4))
    # grid.plot_maps(exp(arrangeT.logpi),cmap='jet',vmax=1,grid=[2,3])
    # cluster = np.argmax(arrangeT.logpi,axis=0)
    # grid.plot_maps(cluster,cmap='tab10',vmax=9,grid=[2,3],offset=6)

    # Step 4: Generate data by sampling from the above model

    p = pt.ones(K)
    U = ar.sample_multinomial(pt.softmax(arrangeT.bu,0),shape=(N,K,grid.P))
    if do_plot>1:
        plt.figure(figsize=(10,4))
    for i in range (10):
        _,H = arrangeT.sample_h(U)
        _,U = arrangeT.sample_U(H)
        if do_plot>1:
            u = ar.compress_mn(U)
            grid.plot_maps(u[8],cmap='tab10',vmax=K,grid=[2,5],offset=i+1)
        # plt.figure(10,4)
        # grid.plot_maps(h[0],cmap='tab10',vmax=K,grid=[2,5],offset=i+1)

    Utrue = ar.compress_mn(U)


    #This is the training data
    Ytrain = MT.emission.sample(Utrue.numpy())
    Ytest = MT.emission.sample(Utrue.numpy())  # Testing data

    # Plot first 10 samples
    if do_plot>0:
        plt.figure(figsize=(13,5))
        grid.plot_maps(Utrue[0:10],cmap='tab10',vmax=K,grid=[2,5])

    return Ytrain, Ytest, U, MT , grid

def eval_arrange(models,emloglik_train,emloglik_test,Utrue):
    D= pd.DataFrame()
    Utrue_mn = ar.expand_mn(Utrue,models[0].K)
    for m in models:
        EU,_ = m.Estep(emloglik_train)
        logpy_test1= ev.logpY(emloglik_test,EU)
        uerr_test1= ev.u_abserr(Utrue_mn,EU)
        dict ={'model':[m.name],
               'type':['test'],
               'uerr':uerr_test1,
               'logpy':logpy_test1}

        D=pd.concat([D,pd.DataFrame(dict)],ignore_index=True)
    return D


def simulation_1():
    K =5
    N = 20
    num_subj=500
    sigma2=0.5
    batch_size=20
    n_epoch=40
    pt.set_default_dtype(pt.float32)

    Ytrain,Ytest,Utrue,Mtrue,grid = make_mrf_data(10,K,N=N,
            num_subj=num_subj,
            theta_mu=20,theta_w=2,sigma2=sigma2,
            do_plot=True)

    emloglik_train = Mtrue.emission.Estep(Y=Ytrain)
    emloglik_test = Mtrue.emission.Estep(Y=Ytest)
    P = Mtrue.emission.P

    # Generate partitions for region-completion testing
    num_part = 4
    p=pt.ones(num_part)/num_part
    part = pt.multinomial(p,P,replacement=True)

    # Independent spatial arrangement model
    indepAr = ar.ArrangeIndependent(K=K,P=P,spatial_specific=True,remove_redundancy=False)
    indepAr.name='idenp'

    # blank restricted bolzman machine
    n_hidden = 30 # 30 hidden nodes
    rbm = ar.mpRBM_pCD(K,P,n_hidden,eneg_iter=3,eneg_numchains=200)
    rbm.name=f'RBM_{n_hidden}'

    # Get the true pott models
    Mpotts = copy.deepcopy(Mtrue.arrange)
    Mpotts.epos_numchains=100
    Mpotts.epos_iter =5

    # Train those two models
    indepAr,T = train_sml(indepAr,
            emloglik_train,emloglik_test,
            part=part,n_epoch=n_epoch,batch_size=N)

    rbm.alpha = 0.001
    rbm.bu=indepAr.logpi.detach().clone()
    rbm.W = pt.randn(rbm.nh,P*K)*0.1

    rbm, T1 = train_sml(rbm,
            emloglik_train,emloglik_test,
            part,batch_size=batch_size,n_epoch=n_epoch)

    T = pd.concat([T,T1],ignore_index=True)

    plt.figure(figsize=(8,8))
    sb.lineplot(data=T[T.iter>3],y='uerr',x='iter',hue='model',style='type')

    # Get the final error and the true pott models
    D = eval_arrange([indepAr,rbm,Mpotts],emloglik_train,emloglik_test,Utrue)

    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    sb.barplot(data=D,x='model',y='uerr')
    plt.subplot(1,2,2)
    sb.barplot(data=D,x='model',y='logpy')
    pass

def simulation_2():
    K =5
    N = 20
    num_subj=500
    sigma2=0.5
    batch_size=20
    n_epoch=40
    pt.set_default_dtype(pt.float32)

    Ytrain,Ytest,Utrue,Mtrue,grid = make_cmpRBM_data(10,K,N=N,
            num_subj=num_subj,
            theta_mu=20,theta_w=1.0,sigma2=sigma2,
            do_plot=1)

    emloglik_train = Mtrue.emission.Estep(Y=Ytrain)
    emloglik_test = Mtrue.emission.Estep(Y=Ytest)
    P = Mtrue.emission.P

    # Generate partitions for region-completion testing
    num_part = 4
    p=pt.ones(num_part)/num_part
    part = pt.multinomial(p,P,replacement=True)

    # Independent spatial arrangement model
    indepAr = ar.ArrangeIndependent(K=K,P=P,spatial_specific=True,remove_redundancy=False)
    indepAr.name='idenp'

    # blank restricted bolzman machine
    n_hidden = 100 # 30 hidden nodes
    rbm1 = ar.mpRBM_pCD(K,P,n_hidden,eneg_iter=3,eneg_numchains=200)
    rbm1.name=f'RBM_{n_hidden}'

    # Convolutional Boltzmann:
    n_hidden = P # hidden nodes
    rbm2 = ar.cmpRBM_pCD(K,P,nh=n_hidden,eneg_iter=3,eneg_numchains=200)
    rbm2.name=f'cRBM_{n_hidden}'

    # Covolutional
    Wc = Mtrue.arrange.Wc
    rbm3 = ar.cmpRBM_pCD(K,P,Wc=Wc,theta=0.1, eneg_iter=3,eneg_numchains=200)
    rbm3.name=f'cRBM_Wc'

    # Get the true pott models
    Mpotts = copy.deepcopy(Mtrue.arrange)
    Mpotts.epos_numchains=100
    Mpotts.epos_iter =5

    # Make list of candidate models
    Models = [indepAr,rbm1,rbm2,rbm3,Mpotts]

    # Train those two models
    indepAr,T = train_sml(indepAr,
            emloglik_train,emloglik_test,
            part=part,n_epoch=n_epoch,batch_size=N)

    for m in Models[1:4]:
        m.alpha = 0.001
        m.bu=indepAr.logpi.detach().clone()

        m, T1 = train_sml(m,
            emloglik_train,emloglik_test,
            part,batch_size=batch_size,n_epoch=n_epoch)

        T = pd.concat([T,T1],ignore_index=True)

    plt.figure(figsize=(8,8))
    sb.lineplot(data=T[T.iter>3],y='logpy',x='iter',hue='model',style='type')

    # Get the final error and the true pott models
    D = eval_arrange(Models,emloglik_train,emloglik_test,Utrue)

    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    sb.barplot(data=D,x='model',y='uerr')
    plt.subplot(1,2,2)
    sb.barplot(data=D,x='model',y='logpy')
    pass


def test_cmpRBM():
    K =5
    N=20
    num_subj=10
    sigma2=0.1
    batch_size=100
    n_epoch=40
    pt.set_default_dtype(pt.float32)

    Ytrain, Ytest, U, MT , grid = make_cmpRBM_data(10,K,N,
            num_subj=num_subj,
            theta_mu=20,theta_w=1,sigma2=sigma2,
            do_plot=True)

    pass



if __name__ == '__main__':
    # compare_gibbs()
    # train_rbm_to_mrf2('notebooks/sim_500.pt',n_hidden=[30,100],batch_size=20,n_epoch=20,sigma2=0.5)
    simulation_2()
    # pass
    # test_cmpRBM()
    # test_sample_multinomial()
    # train_RBM()
