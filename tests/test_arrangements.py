import os  # to handle path information
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
from numpy import exp,log,sqrt
import torch as pt
import arrangements as ar
import emissions as em
import full_model as fm
import spatial as sp
import copy
import evaluation as ev

def train_rbm(rbm,emlog_train,emlog_test,part,crit='logpY',
             n_epoch=20,batch_size=20,alpha=0.05):
    N = emlog_train.shape[0]
    Utrain=pt.softmax(emlog_train,dim=1)
    uerr_train = np.zeros(n_epoch)
    uerr_test1 = np.zeros(n_epoch)
    uerr_test2 = np.zeros(n_epoch)
    # Intialize negative sampling
    if rbm.train == 'pCD':
        rbm.eneg_pCD(num_chains=rbm.eneg_numchains,iter=20)
    for epoch in range(n_epoch):
        # Get test error
        EU,_ = rbm.Estep(emlog_train,gather_ss=False)
        pi = rbm.get_pi(Utrain)
        uerr_train[epoch] = ev.evaluate_full_arr(emlog_train,pi,crit=crit)
        uerr_test1[epoch]= ev.evaluate_full_arr(emlog_test,EU,crit=crit)
        uerr_test2[epoch]= ev.evaluate_completion_arr(rbm,emlog_test,part,crit=crit)
        print(f'epoch {epoch:2d} Train: {uerr_train[epoch]:.4f}, Test1: {uerr_test1[epoch]:.4f}, Test2: {uerr_test2[epoch]:.4f}')

        for b in range(0,N-batch_size+1,batch_size):
            ind = range(b,b+batch_size)
            rbm.Estep(emlog_train[ind,:,:])
            u,Eh = rbm.Eneg(Utrain[ind,:,:])
            rbm.Mstep(alpha=alpha)
    return rbm,uerr_train,uerr_test1,uerr_test2

def train_ind(indepAr,emlog_train,emlog_test,part,crit='logpY',n_epoch=20):
    uerr_train = np.zeros(n_epoch)
    uerr_test1 = np.zeros(n_epoch)
    uerr_test2 = np.zeros(n_epoch)

    for epoch in range(n_epoch):
        EU,ll = indepAr.Estep(emlog_train)
        pi = pt.softmax(indepAr.logpi,dim=0)
        uerr_train[epoch] = ev.evaluate_full_arr(emlog_train,pi,crit=crit)
        uerr_test1[epoch] = ev.evaluate_full_arr(emlog_train,EU,crit=crit)
        uerr_test2[epoch] = ev.evaluate_completion_arr(indepAr,emlog_test,part,crit=crit)

        print(f'epoch {epoch:2d}, ll: {ll.sum():.1f}, Train: {uerr_train[epoch]:.4f}, Test1: {uerr_test1[epoch]:.4f}')
        indepAr.Mstep()
    pass
    return indepAr, uerr_train, uerr_test1, uerr_test2



def train_rbm_to_mrf(width=10,K=5,N=200,theta_mu=20,theta_w=2,sigma2=0.5,
                    n_hidden = 30,n_epoch=20, batch_size=20,alpha=0.01):
    """Fits and RBM to observed activiy data, given a smooth true arrangement (mrf)
    Current version keeps the emission model stable and known

    Args:
        width (int): [description]. Defaults to 10.
        K (int): [description]. Defaults to 5.
        N (int): [description]. Defaults to 200.
        theta_mu (int): [description]. Defaults to 20.
        theta_w (int): [description]. Defaults to 2.
        sigma2 (float): [description]. Defaults to 0.5.
        n_hidden (int): [description]. Defaults to 100.
        n_epoch (int): [description]. Defaults to 20.
        batch_size (int): size of each learning batch

    Returns:
        [type]: [description]
    """

    lossU = 'logpY'
    # Step 1: Create the true model
    grid = sp.SpatialGrid(width=width,height=width)
    arrangeT = ar.PottsModel(grid.W, K=K)
    emissionT = em.MixGaussian(K=K, N=N, P=grid.P)
    P = arrangeT.P

    # Step 2: Initialize the parameters of the true model
    arrangeT.random_smooth_pi(grid.Dist,theta_mu=theta_mu)
    arrangeT.theta_w = theta_w
    emissionT.random_params()
    emissionT.sigma2=sigma2
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
    plt.figure(figsize=(10,4))
    grid.plot_maps(U[0:10],cmap='tab10',vmax=9,grid=[2,5])

    # Step 5: Generate partitions for region-completion testing
    num_part = 4
    p=pt.ones(num_part)/num_part
    part = pt.multinomial(p,arrangeT.P,replacement=True)

    # Make two versions of the model for fitting
    rbm1 = ar.mpRBM(K=K,P=P,nh=n_hidden)
    rbm1.Etype='vis'
    rbm1.train = 'pCD'
    rbm1.eneg_iter = 3
    rbm1.eneg_numchains = 77

    rbm2 = copy.deepcopy(rbm1)
    rbm2.Etype='prob'
    rbm2.train = 'pCD'
    rbm2.eneg_iter = 3
    rbm2.eneg_numchains = 77

    emloglik_train = MT.emission.Estep(Y=Ytrain)
    emloglik_test = MT.emission.Estep(Y=Ytest)
    emloglik_train=pt.tensor(emloglik_train,dtype=pt.get_default_dtype())
    emloglik_test=pt.tensor(emloglik_test,dtype=pt.get_default_dtype())

    # Check the baseline for independent Arrangement model
    indepAr = ar.ArrangeIndependent(K=K,P=P,spatial_specific=True,remove_redundancy=False)

    indepAr,uerr_tr3,uerr_t3,uerr_c3 = train_ind(indepAr,
            emloglik_train,emloglik_test,
            part=part,n_epoch=n_epoch)

    # Test normal training of RBM using CDk=1
    rbm, uerr_tr1,uerr_t1,uerr_c1 = train_rbm(rbm1,
            emloglik_train,emloglik_test,
            part,batch_size=batch_size,n_epoch=n_epoch,alpha=alpha)

    # Test training using persistent CD
    rbm2.bu=indepAr.logpi.copy()

    rbm2, uerr_tr2,uerr_t2,uerr_c2 = train_rbm(rbm2,
            emloglik_train,emloglik_test,
            part,batch_size=batch_size,n_epoch=n_epoch,alpha=alpha)


    t=np.arange(0,n_epoch)
    plt.figure(figsize=(6,8))
    plt.plot(t,uerr_tr1,'r',label='training')
    plt.plot(t,uerr_t1,'r--',label='test1')
    plt.plot(t,uerr_c1,'r:',label='test2')
    plt.plot(t,uerr_tr2,'b',label='training')
    plt.plot(t,uerr_t2,'b--',label='test1')
    plt.plot(t,uerr_c2,'b:',label='test2')
    plt.plot(t,uerr_tr3,'g',label='training')
    plt.plot(t,uerr_t3,'g--',label='test1')
    plt.plot(t,uerr_c3,'g:',label='test1')

    plt.figure(figsize=(6,8))
    Utrue=ar.expand_mn(U,K)
    pi = Utrue.mean(dim=0)
    piE = pt.softmax(emloglik_train,dim=1).mean(dim=0)
    pi1 = rbm.eneg_U.mean(dim=0)
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

if __name__ == '__main__':
    train_rbm_to_mrf(N=60,batch_size=20,n_epoch=60,sigma2=0.01)
    # train_RBM()