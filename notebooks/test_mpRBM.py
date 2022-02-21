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

def train_sml(model,emlog_train,emlog_test,part,crit='logpY',
             n_epoch=20,batch_size=20,verbose=False):
    N = emlog_train.shape[0]
    Utrain=pt.softmax(emlog_train,dim=1)
    uerr_train = np.zeros(n_epoch)
    uerr_test1 = np.zeros(n_epoch)
    uerr_test2 = np.zeros(n_epoch)
    # Intialize negative sampling
    for epoch in range(n_epoch):
        # Get test error
        EU,_ = model.Estep(emlog_train,gather_ss=False)
        pi = model.marginal_prob(Utrain)
        uerr_train[epoch] = ev.evaluate_full_arr(emlog_train,pi,crit=crit)
        uerr_test1[epoch]= ev.evaluate_full_arr(emlog_test,EU,crit=crit)
        uerr_test2[epoch]= ev.evaluate_completion_arr(model,emlog_test,part,crit=crit)
        if (verbose):
            print(f'epoch {epoch:2d} Train: {uerr_train[epoch]:.4f}, Test1: {uerr_test1[epoch]:.4f}, Test2: {uerr_test2[epoch]:.4f}')

        # Update the model in batches
        for b in range(0,N-batch_size+1,batch_size):
            ind = range(b,b+batch_size)
            model.Estep(emlog_train[ind,:,:])
            model.Eneg(Utrain[ind,:,:])
            model.Mstep()

    # Make a data frame for the results
    T1 = pd.DataFrame({'model':[model.name]*n_epoch,
                        'type':['train']*n_epoch,
                        'iter':np.arange(n_epoch),
                        'uerr':uerr_train})
    T2 = pd.DataFrame({'model':[model.name]*n_epoch,
                        'type':['test']*n_epoch,
                        'iter':np.arange(n_epoch),
                        'uerr':uerr_test1})
    T3 = pd.DataFrame({'model':[model.name]*n_epoch,
                        'type':['compl']*n_epoch,
                        'iter':np.arange(n_epoch),
                        'uerr':uerr_test2})

    T = pd.concat([T1,T2,T3],ignore_index=True)
    return model,T

def make_mrf_data(width=10,K=5,N=200,
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
    emissionT = em.MixGaussian(K=K, N=N, P=grid.P)

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

def train_rbm_to_mrf(N=200,n_hidden = 30,n_epoch=20, batch_size=20, sigma2=0.01):
    """Fits and RBM to observed activiy data, given a smooth true arrangement (mrf)
    Current version keeps the emission model stable and known

    Args:
        n_hidden (int): [description]. Defaults to 100.
        n_epoch (int): [description]. Defaults to 20.
        batch_size (int): size of each learning batch

    Returns:
        [type]: [description]
    """
    K =5
    Ytrain,Ytest,Utrue,Mtrue =  make_mrf_data(10,K,N,
            theta_mu=20,theta_w=2,sigma2=sigma2,
            do_plot=True)
    lossU = 'logpY'
    P = Mtrue.arrange.P

    # Step 5: Generate partitions for region-completion testing
    num_part = 4
    p=pt.ones(num_part)/num_part
    part = pt.multinomial(p,P,replacement=True)

    # Make two versions of the model for fitting
    rbm1 = ar.mpRBM_pCD(K,P,n_hidden,eneg_iter=3,eneg_numchains=77)
    rbm1.Etype='vis'
    rbm1.name = 'vis'

    rbm2 = ar.mpRBM_pCD(K,P,n_hidden,eneg_iter=3,eneg_numchains=77)
    rbm2.Etype='prob'
    rbm2.name = 'prob'

    emloglik_train = Mtrue.emission.Estep(Y=Ytrain)
    emloglik_test = Mtrue.emission.Estep(Y=Ytest)

    # Check the baseline for independent Arrangement model
    indepAr = ar.ArrangeIndependent(K=K,P=P,spatial_specific=True,remove_redundancy=False)
    indepAr.name = 'indep'

    indepAr,T1 = train_sml(indepAr,
            emloglik_train,emloglik_test,
            part=part,n_epoch=n_epoch,batch_size=N)

    # Test normal training of RBM using CDk=1
    rbm1, T2 = train_sml(rbm1,
            emloglik_train,emloglik_test,
            part,batch_size=batch_size,n_epoch=n_epoch)

    # Test training using persistent CD

    rbm2, T3 = train_sml(rbm2,
            emloglik_train,emloglik_test,
            part,batch_size=batch_size,n_epoch=n_epoch)

    T = pd.concat([T1,T2,T3],ignore_index=True)

    plt.figure(figsize=(8,8))
    sb.lineplot(data=T,y='uerr',x='iter',hue='model',style='type')

    plt.figure(figsize=(6,8))
    Utrue=ar.expand_mn(Utrue,K)
    pi = Utrue.mean(dim=0)
    piE = pt.softmax(emloglik_train,dim=1).mean(dim=0)
    pi1 = rbm1.eneg_U.mean(dim=0)
    pi2 = rbm2.eneg_U.mean(dim=0)
    pi3 = pt.softmax(indepAr.logpi,dim=0)

    plt.subplot(2,2,1)
    plt.scatter(pi,pi1)
    plt.subplot(2,2,2)
    plt.scatter(pi,pi2)
    plt.subplot(2,2,3)
    plt.scatter(pi,pi3)
    plt.subplot(2,2,4)
    plt.scatter(pi,piE)

    pass

def train_rbm_to_mrf2(N=200,
    n_hidden = [30,100],
    n_epoch=20,
    batch_size=20,
    sigma2=0.01):
    """Fits and RBM to observed activiy data, given a smooth true arrangement (mrf)
    Current version keeps the emission model stable and known

    Args:
        n_hidden (int): [description]. Defaults to 100.
        n_epoch (int): [description]. Defaults to 20.
        batch_size (int): size of each learning batch

    Returns:
        [type]: [description]
    """
    K =5
    if type(N) is str:
        [Utrue,Mtrue]=pt.load(N)
        Mtrue.emission.sigma2=pt.tensor(sigma2)
        Ytrain = Mtrue.emission.sample(Utrue) # This is the training data
        Ytest = Mtrue.emission.sample(Utrue)  # Testing data
        N = Utrue.shape[0]
    else:
        Ytrain,Ytest,Utrue,Mtrue =  make_mrf_data(10,K,N,
            theta_mu=20,theta_w=2,sigma2=sigma2,
            do_plot=True)
    lossU = 'logpY'
    P = Mtrue.arrange.P

    # Step 5: Generate partitions for region-completion testing
    num_part = 4
    p=pt.ones(num_part)/num_part
    part = pt.multinomial(p,P,replacement=True)

    indepAr = ar.ArrangeIndependent(K=K,P=P,spatial_specific=True,remove_redundancy=False)
    indepAr.name='idenp'

    rbm1 = ar.mpRBM_pCD(K,P,n_hidden[0],eneg_iter=3,eneg_numchains=200)
    rbm1.name='RBM1'
    rbm1.alpha = 0.001

    rbm2 = ar.mpRBM_pCD(K,P,n_hidden[1],eneg_iter=3,eneg_numchains=200)
    rbm2.name='RBM2'
    rbm2.alpha = 0.001

    emloglik_train = Mtrue.emission.Estep(Y=Ytrain)
    emloglik_test = Mtrue.emission.Estep(Y=Ytest)

    # Check the baseline for independent Arrangement model

    indepAr,T1 = train_sml(indepAr,
            emloglik_train,emloglik_test,
            part=part,n_epoch=n_epoch,batch_size=N)

    rbm1.bu=indepAr.logpi.detach().clone()
    rbm1.W = pt.randn(n_hidden[0],P*K)*0.1

    rbm1, T2 = train_sml(rbm1,
            emloglik_train,emloglik_test,
            part,batch_size=batch_size,n_epoch=n_epoch)


    rbm2.bu=indepAr.logpi.detach().clone()
    rbm2.W = pt.randn(n_hidden[1],P*K)*0.1

    rbm2, T3 = train_sml(rbm2,
            emloglik_train,emloglik_test,
            part,batch_size=batch_size,n_epoch=n_epoch)

    T = pd.concat([T1,T2,T3],ignore_index=True)

    plt.figure(figsize=(8,8))
    sb.lineplot(data=T,y='uerr',x='iter',hue='model',style='type')

    grid=sp.SpatialGrid(10,10)
    plt.figure(figsize=(8,8))
    grid.plot_maps(Utrue[0:5],grid=(3,5))
    uin = indepAr.sample(5)
    grid.plot_maps(uin[0:5],grid=(3,5),offset=6)
    urbm = rbm1.sample(5,iter=20)
    grid.plot_maps(urbm[0:5],grid=(3,5),offset=11)

    pass

def test_epos_meanfield(n_hidden = 100,
    n_epoch=20,
    batch_size=20,
    sigma2=0.5):
    """Checks different number of iterations for the positive Estep
    Checks speed of convergence for RBM_pCD model
    Args:
        n_hidden (int): [description]. Defaults to 100.
        n_epoch (int): [description]. Defaults to 20.
        batch_size (int): size of each learning batch

    Returns:
        [type]: [description]
    """
    [Utrue,Mtrue]=pt.load('notebooks/sim_500.pt')
    Mtrue.emission.sigma2=pt.tensor(sigma2)
    Ytrain = Mtrue.emission.sample(Utrue) # This is the training data
    Ytest = Mtrue.emission.sample(Utrue)  # Testing data
    K = Mtrue.emission.K
    N = Utrue.shape[0]
    P = Mtrue.arrange.P
    lossU = 'logpY'

    # Step 5: Generate partitions for region-completion testing
    num_part = 4
    p=pt.ones(num_part)/num_part
    part = pt.multinomial(p,P,replacement=True)

    # Build different models
    indepAr = ar.ArrangeIndependent(K=K,P=P,spatial_specific=True,remove_redundancy=False)
    indepAr.name='idenp'

    rbm1 = ar.mpRBM_pCD(K,P,n_hidden,eneg_iter=3,eneg_numchains=200)
    rbm1.epos_iter=1
    rbm1.name='RBM1'
    rbm1.alpha = 0.001

    rbm2 = ar.mpRBM_pCD(K,P,n_hidden,eneg_iter=2,eneg_numchains=200)
    rbm2.epos_iter=1
    rbm2.name='RBM2'
    rbm2.alpha = 0.001

    emloglik_train = Mtrue.emission.Estep(Y=Ytrain)
    emloglik_test = Mtrue.emission.Estep(Y=Ytest)

    # Check the baseline for independent Arrangement model

    t = time.time()
    indepAr,T1 = train_sml(indepAr,
            emloglik_train,emloglik_test,
            part=part,n_epoch=n_epoch,batch_size=N)
    print(f"Indep_time:{time.time()-t:.3f}")

    rbm1.bu=indepAr.logpi.detach().clone()
    rbm2.bu=indepAr.logpi.detach().clone()
    rbm1.W = pt.randn(n_hidden,P*K)*0.1
    rbm2.W = rbm1.W.detach().clone()

    t = time.time()
    rbm1, T2 = train_sml(rbm1,
            emloglik_train,emloglik_test,
            part,batch_size=batch_size,n_epoch=n_epoch)
    print(f"RBM1_time:{time.time()-t:.3f}")
    t = time.time()
    rbm2, T3 = train_sml(rbm2,
            emloglik_train,emloglik_test,
            part,batch_size=batch_size,n_epoch=n_epoch)
    print(f"RBM2_time:{time.time()-t:.3f}")

    T = pd.concat([T1,T2,T3],ignore_index=True)

    plt.figure(figsize=(8,8))
    sb.lineplot(data=T,y='uerr',x='iter',hue='model',style='type')


    pass

def compare_gibbs():
    """Compares different implementations of Gibbs sampling
    """
    [Utrue,Mtrue]=pt.load('notebooks/sim_500.pt')
    M = Mtrue.arrange
    t = time.time()
    M.calculate_neighbours()
    print(f"Neigh:{time.time()-t:.3f}")
    t = time.time()
    U1 = M.sample_gibbs(num_chains=100,bias=M.logpi,iter=5)
    print(f"time 1:{time.time()-t:.3f}")
    t = time.time()
    U2 = M.sample_gibbs2(num_chains=100,bias=M.logpi,iter=5)
    print(f"time 2:{time.time()-t:.3f}")
    pass

def test_sample_multinomial():
    """ Do some unit test of sample multinomial Function
    """
    p = pt.empty((100,5,200)).uniform_(0,1)
    p = pt.softmax(p,dim=1)
    t = time.time()
    U1 = ar.sample_multinomial_old(p,compress=True)
    print(f"time 1:{time.time()-t:.5f}")
    t = time.time()
    U2 = ar.sample_multinomial(p,kdim=1,compress=True)
    print(f"time 2:{time.time()-t:.5f}")
    pass
    pass
    U2 = ar.sample_multinomial(p[0,:,0:1],(2,4,10))
    U3 = ar.sample_multinomial(p[0,:,0:1],(5,10))
    p = pt.tensor([0.1,0.2,0.7])
    U4 = ar.sample_multinomial(p.reshape(3,1),(5,3,10000))
    # Check:
    U4.mean(dim=(0,2))
    pass



if __name__ == '__main__':
    # compare_gibbs()
    # train_rbm_to_mrf2('notebooks/sim_500.pt',n_hidden=[30,100],batch_size=20,n_epoch=20,sigma2=0.5)
    test_epos_meanfield()
    # test_sample_multinomial()
    # train_RBM()
