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
from test_mpRBM import train_sml,make_mrf_data
import pandas as pd
import seaborn as sb
import copy


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


def simulation(): 
    K =5
    N=500
    sigma2=0.1
    batch_size=100 
    n_epoch=40
    pt.set_default_dtype(pt.float32)
    
    Ytrain,Ytest,Utrue,Mtrue,grid = make_mrf_data(10,K,N,
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

if __name__ == '__main__':
    # compare_gibbs()
    # train_rbm_to_mrf2('notebooks/sim_500.pt',n_hidden=[30,100],batch_size=20,n_epoch=20,sigma2=0.5)
    simulation()
    pass
    # test_sample_multinomial()
    # train_RBM()
