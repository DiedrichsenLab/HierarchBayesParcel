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
            theta_mu=20,theta_w=1.0,emission_model=None,
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
    P = width*width
    # Step 1: Create the true model
    grid = sp.SpatialGrid(width=width,height=width)
    W = grid.get_neighbour_connectivity()
    W += pt.eye(W.shape[0])

    # Step 2: Initialize the parameters of the true model
    arrangeT = ar.cmpRBM(K,grid.P,Wc=W,theta=theta_w)
    arrangeT.name = 'cmpRDM'
    arrangeT.bu = grid.random_smooth_pi(K=K,theta_mu=theta_mu,
            centroids=[0,width-1,int(P/2+width/2),P-width,P-1])

    MT = fm.FullModel(arrangeT,emission_model)

    # grid.plot_maps(pt.softmax(arrangeT.bu,0),cmap='hot',vmax=1,grid=[1,5])

    # Step 3: Plot the prior of the true mode
    # plt.figure(figsize=(7,4))
    # grid.plot_maps(exp(arrangeT.logpi),cmap='jet',vmax=1,grid=[2,3])
    # cluster = np.argmax(arrangeT.logpi,axis=0)
    # grid.plot_maps(cluster,cmap='tab10',vmax=9,grid=[2,3],offset=6)

    # Step 4: Generate data by sampling from the above model

    p = pt.ones(K)
    U = ar.sample_multinomial(pt.softmax(arrangeT.bu,0),shape=(num_subj,K,grid.P))
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
    Ytrain = MT.emission.sample(Utrue)
    Ytest = MT.emission.sample(Utrue)  # Testing data

    # Plot first 10 samples
    if do_plot>0:
        plt.figure(figsize=(13,5))
        grid.plot_maps(Utrue[0:10],cmap='tab10',vmax=K,grid=[2,5])

    return Ytrain, Ytest, Utrue, MT , grid

def make_cmpRBM_chain(P=5,K=3,num_subj=20,
            theta_w=1.0,emission_model=None,logpi=2):
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
    grid = sp.SpatialChain(P=5)
    W = grid.get_neighbour_connectivity()
    W += pt.eye(W.shape[0])

    # Step 2: Initialize the parameters of the true model: Only ends are fixed
    arrangeT = ar.cmpRBM(K,grid.P,Wc=W,theta=theta_w)
    arrangeT.name = 'cmpRDM'
    arrangeT.bu = pt.zeros((K,P))
    arrangeT.bu[0,0]=logpi
    arrangeT.bu[-1,-1]=logpi


    MT = fm.FullModel(arrangeT,emission_model)
    # Step 4: Generate data by sampling from the above model
    Utrue = MT.arrange.sample(num_subj,50)
    Ytrain = MT.emission.sample(Utrue)
    Ytest = MT.emission.sample(Utrue)  # Testing data

    return Ytrain, Ytest, Utrue, MT , grid



def train_sml(arM,emM,Ytrain,Ytest,part,crit='Ecos_err',
             n_epoch=20,batch_size=20,verbose=False):
    """Trains only arrangement model, given a fixed emission
    likelhoood.

    Args:
        arM (ArrangementMode): 
        emM (EmissionModel )
        Y_train (tensor): Y_testing log likelihood (KxP)
        Y_test (tensor): Y_training log likelihood test (KxP)
        part (tensor): 1xP partition number for completion test
        crit (str): _description_. Defaults to 'logpY'.
        n_epoch (int): _description_. Defaults to 20.
        batch_size (int): _description_. Defaults to 20.
        verbose (bool): _description_. Defaults to False.

    Returns:
        model: Fitted model
        T: Pandas data frame with epoch level performance metrics
        thetaH: History of fitted thetas
    """
    emlog_train = emM.Estep(Ytrain)
    emlog_test = emM.Estep(Ytest)
    num_subj = emlog_train.shape[0]
    Utrain=pt.softmax(emlog_train,dim=1)

    crit_types = ['train','marg','test','compl'] # different evaluation types
    CR = np.zeros((len(crit_types),n_epoch))
    theta_hist = np.zeros((arM.nparams,n_epoch))
    # Intialize negative sampling
    for epoch in range(n_epoch):
        # Get test error
        EU,_ = arM.Estep(emlog_train,gather_ss=False)
        for i, ct in enumerate(crit_types):
            # Training emission logliklihood:
            if ct=='train':
                CR[i,epoch] = ev.evaluate_full_arr(emM,Ytrain,EU,crit=crit)
            elif ct=='marg':
                pi = arM.marginal_prob()
                CR[i,epoch] = ev.evaluate_full_arr(emM,Ytest,pi,crit=crit)
            elif ct=='test':
                CR[i,epoch] = ev.evaluate_full_arr(emM,Ytest,EU,crit=crit)
            elif ct=='compl':
                CR[i,epoch] = ev.evaluate_completion_arr(arM,emM,Ytest,part,crit=crit)
        if (verbose):
            print(f'epoch {epoch:2d} Test: {crit[2,epoch]:.4f}')

        # Update the model in batches
        for b in range(0,num_subj-batch_size+1,batch_size):
            ind = range(b,b+batch_size)
            arM.Estep(emlog_train[ind,:,:])
            if hasattr(arM,'Eneg'):
                arM.Eneg(use_chains=ind,
                           emission_model=emM)
            arM.Mstep()

        # Record the parameters
        theta_hist[:,epoch]=arM.get_params()

    # Make a data frame for the results
    T=pd.DataFrame()
    for i, ct in enumerate(crit_types):
        T1 = pd.DataFrame({'model':[arM.name]*n_epoch,
                        'type':[ct]*n_epoch,
                        'iter':np.arange(n_epoch),
                        'crit':CR[i]})
        T = pd.concat([T,T1],ignore_index=True)
    return arM,T,theta_hist

def eval_arrange(models,emM,Ytrain,Ytest,Utrue):
    D= pd.DataFrame()
    Utrue_mn = ar.expand_mn(Utrue,emM.K)
    emloglik_train = emM.Estep(Ytrain)
    
    for m in models:
        if m=='data':
            EU = pt.softmax(emloglik_train,1)
            name = m 
        elif m=='Utrue':
            EU = Utrue_mn
            name = m 
        else: 
            EU,_ = m.Estep(emloglik_train)
            name = m.name
        uerr_test1= ev.u_abserr(Utrue_mn,EU)
        cos_err= ev.coserr(Ytest,emM.V,EU,adjusted=False,
                 soft_assign=False).mean(dim=0).item()
        Ecos_err= ev.coserr(Ytest,emM.V,EU,adjusted=False,
                 soft_assign=True).mean(dim=0).item()

        dict ={'model':[name],
               'type':['test'],
               'uerr':uerr_test1,
               'cos_err':cos_err,
               'Ecos_err':Ecos_err}
        D=pd.concat([D,pd.DataFrame(dict)],ignore_index=True)
    return D

def eval_arrange_compl(models,emM,Y,part,Utrue):
    D= pd.DataFrame()
    Utrue_mn = ar.expand_mn(Utrue,models[0].K)
    for m in models:
        cos_err_compl = ev.evaluate_completion_arr(m,emM,Y,part,crit='cos_err')
        Ecos_err_compl = ev.evaluate_completion_arr(m,emM,Y,part,crit='Ecos_err')
        uerr_compl = ev.evaluate_completion_arr(m,emM,Y,part,
                                    crit='u_abserr',Utrue=Utrue_mn)
        dict ={'model':[m.name],
               'type':['compl'],
               'uerr':uerr_compl,
               'cos_err':cos_err_compl,
               'Ecos_err':Ecos_err_compl}
        D=pd.concat([D,pd.DataFrame(dict)],ignore_index=True)
    # get the baseline for Utrue
    cos_err= ev.coserr(Y,emM.V,Utrue_mn,adjusted=False,
                 soft_assign=False).mean(dim=0).item()
    Ecos_err= ev.coserr(Y,emM.V,Utrue_mn,adjusted=False,
                 soft_assign=True).mean(dim=0).item()
    dict ={'model':['Utrue'],
               'type':['compl'],
               'uerr':0,
               'cos_err':cos_err,
               'Ecos_err':Ecos_err}
    D=pd.concat([D,pd.DataFrame(dict)],ignore_index=True)
    return D

def plot_Uhat_maps(models,emloglik,grid):
    plt.figure(figsize=(10,7))
    n_models = len(models)
    K = emloglik.shape[1]
    for i,m in enumerate(models):
        if m is None:
            Uh=pt.softmax(emloglik,dim=1)
        else:
            Uh,_ = m.Estep(emloglik)
        grid.plot_maps(Uh[0],cmap='jet',vmax=1,grid=(n_models,K),offset=i*K+1)

def plot_P_maps(pmaps,grid):
    plt.figure(figsize=(10,7))
    n_models = len(pmaps)
    K = pmaps[0].shape[0]
    for i,m in enumerate(pmaps):
        grid.plot_maps(m,cmap='jet',vmax=1,grid=(n_models,K),offset=i*K+1)

def plot_individual_Uhat(models,Utrue, emloglik,grid,style='prob'):
    # Get the expectation
    n_models = len(models)+2 
    K = emloglik.shape[1]
    P = emloglik.shape[2]
    
    Uh = [] 
    plt.figure(figsize=(10,7))
    Uh.append(ar.expand_mn(Utrue[0:1],K))
    Uh.append(pt.softmax(emloglik[0:1],dim=1))
    for i,m in enumerate(models):
        A,_=m.Estep(emloglik[0:1])
        Uh.append(A)

    if style=='prob':
        for i,uh in enumerate(Uh): 
            grid.plot_maps(uh[0],cmap='jet',vmax=1,
                    grid=(n_models,K),
                    offset = K*i+1)
    elif style=='argmax': 
        ArgM = pt.zeros(n_models,P)
        for i,uh in enumerate(Uh): 
            ArgM[i,:] = pt.argmax(uh[0],dim=0)
        grid.plot_maps(ArgM,cmap='tab10',vmax=K,
                    grid=(1,n_models))
    elif style=='mixed': 
        ArgM = pt.zeros(n_models,P)
        Prob = pt.zeros(n_models,P)

        for i,uh in enumerate(Uh): 
            ArgM[i,:] = pt.argmax(uh[0],dim=0)
            Prob[i,:] = uh[0][2,:]
        grid.plot_maps(ArgM,cmap='tab10',vmax=K,
                    grid=(2,n_models))
        grid.plot_maps(Prob,cmap='jet',vmax=1,
                    grid=(2,n_models),
                    offset = n_models+1)

def plot_evaluation(D,criteria=['uerr','cos_err','Ecos_err']
                      ,types=['test','compl']): 
    # Get the final error and the true pott models
    plt.figure(figsize=(12,7))
    ncrit = len(criteria)
    ntypes = len(types)
    for j in range(ntypes): 
        for i in range(ncrit): 
            plt.subplot(ntypes,ncrit,i+j*ncrit+1)
            sb.barplot(data=D[D.type==types[j]],x='model',y=criteria[i])
            plt.title(f'{criteria[i]}{types[j]}')

def plot_evaluation2(): 
    # Get the final error and the true pott models
    plt.figure(figsize=(3,4))
    D = pd.read_csv('deepMRF.csv')
    T = D[(D.type=='test') & (D.model!='true') & (D.model!='Utrue')]
    noisefloor = D[(D.type=='test') & (D.model=='Utrue')].Ecos_err.mean()
    sb.barplot(data=T,x='model',y='Ecos_err')
    plt.ylim([0.5,0.8])
    plt.axhline(noisefloor)
    pass

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
    n_hidden = 100 # hidden nodes
    rbm = ar.mpRBM_pCD(K,P,n_hidden,eneg_iter=3,eneg_numchains=200)
    rbm.name=f'RBM_{n_hidden}'

    # Get the true pott models
    Mpotts = copy.deepcopy(Mtrue.arrange)
    Mpotts.epos_numchains=100
    Mpotts.epos_iter =5

    # Train the independent arrangement model
    indepAr,T = train_sml(indepAr,
            emloglik_train,emloglik_test,
            part=part,n_epoch=n_epoch,batch_size=num_subj)

    rbm.alpha = 0.001
    rbm.bu=indepAr.logpi.detach().clone()
    rbm.W = pt.randn(rbm.nh,P*K)*0.1

    rbm, T1 = train_sml(rbm,
            emloglik_train,emloglik_test,
            part,batch_size=batch_size,n_epoch=n_epoch)

    T = pd.concat([T,T1],ignore_index=True)

    plt.figure(figsize=(8,8))
    sb.lineplot(data=T[T.iter>0],y='crit',x='iter',hue='model',style='type')

    # Get the final error and the true pott models
    D = eval_arrange([indepAr,rbm,Mpotts],emloglik_train,emloglik_test,Utrue)

    plt.figure(figsize=(8,3))
    plt.subplot(2,2,1)
    sb.barplot(data=D,x='model',y='uerr')
    plt.subplot(2,2,2)
    sb.barplot(data=D,x='model',y='logpy')

    plot_Uhat_maps([None,indepAr,rbm,Mpotts],emloglik_test[0:1],grid)
    pass

def simulation_2():
    K = 5
    width = 30
    P = width * width
    theta_mu = P / 2
    num_subj=30
    batch_size=30
    n_epoch=80
    theta = 1.3
    # Multinomial 
    w = 2.0
    # MixGaussian 
    sigma2 = 0.4
    N = 10

    eneg_iter = 10
    epos_iter = 10
    num_sim = 20 
    
    
    pt.set_default_dtype(pt.float32)
    TT=pd.DataFrame()
    DD=pd.DataFrame()
    HH = np.zeros((num_sim,n_epoch))
    # REcorded bias parameter 
    RecBu = pt.zeros((num_sim,K,P))
    RecLp = pt.zeros((num_sim,K,P))
    RecUtrue = pt.zeros((num_sim,K,P))
    RecEmLog = pt.zeros((num_sim,K,P))

    # Make a new emission model for the simulation
    # emissionM = em.MultiNomial(K=K, P=P)
    # emissionM.w = pt.tensor(w)
    emissionM = em.MixGaussian(K,N,P)
    emissionM.sigma2 = pt.tensor(sigma2)


    for s in range(num_sim):
        Ytrain,Ytest,Utrue,Mtrue,grid = make_cmpRBM_data(width,K,N=N,
            num_subj=num_subj,
            theta_mu=theta_mu,
            theta_w=theta,
            emission_model=emissionM,
            do_plot=0)
        # Ytrain,Ytest,Utrue,Mtrue,grid = make_mrf_data(10,K,N=N,
        #         num_subj=num_subj,
        #         theta_mu=20,theta_w=2,sigma2=sigma2,
        #         do_plot=1)

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

        # Train the independent model as baseline
        indepAr,T,theta1 = train_sml(indepAr,Mtrue.emission,Ytrain,Ytest,
                part=part,n_epoch=n_epoch,batch_size=num_subj)

        # Gte the true arrangement model 
        rbm = Mtrue.arrange
        rbm.name = 'true'

        # Convolutional Boltzmann:
        n_hidden = P # hidden nodes
        rbm2 = ar.cmpRBM(K,P,nh=n_hidden,
                            eneg_iter=eneg_iter,
                            epos_iter=epos_iter,
                            eneg_numchains=num_subj)
        rbm2.name=f'cRBM_{n_hidden}'
        rbm2.W = pt.randn(n_hidden,P)*0.1
        # rbm2.W = Mtrue.arrange.W.detach().clone()
        rbm2.bu= indepAr.logpi.detach().clone()
        # rbm2.bu=  Mtrue.arrange.bu.detach().clone()
        rbm2.alpha = 1
        rbm2.fit_W = True
        rbm2.fit_bu = True

        # Covolutional
        rbm3 = ar.cmpRBM(K,P,Wc=rbm.Wc,
                            theta=theta,
                            eneg_iter=eneg_iter,
                            epos_iter=epos_iter,
                            eneg_numchains=num_subj)
        rbm3.bu = pt.zeros((K,P))
        rbm3.bu = indepAr.logpi.detach().clone()
        # rbm3.bu = rbm.bu.detach().clone()
        rbm3.name=f'cRBM_Wc'
        rbm3.fit_W = False
        rbm3.fit_bu = True
        rbm3.alpha = 1

        # Get the true pott models
        # Mpotts = copy.deepcopy(Mtrue.arrange)
        # Mpotts.epos_numchains=100
        # Mpotts.epos_iter =5

        # Make list of candidate models
        Models = [indepAr,rbm3,rbm]


        TH = [theta1]
        for m in Models[1:2]:

            m, T1,theta_hist = train_sml(m,Mtrue.emission,Ytrain,Ytest,part,
                batch_size=batch_size,
                n_epoch=n_epoch)
            TH.append(theta_hist)
            T = pd.concat([T,T1],ignore_index=True)

        # Evaluate overall
        D = eval_arrange(['data',indepAr,rbm3,rbm,'Utrue'],Mtrue.emission,Ytrain,Ytest,Utrue=Utrue)
        D1 = eval_arrange_compl(Models,Mtrue.emission,Ytest,part=part,Utrue=Utrue)

        DD = pd.concat([DD,D,D1],ignore_index=True)
        TT = pd.concat([TT,T],ignore_index=True)
        HH[s,:]= TH[1][rbm3.get_param_indices('theta'),:]
        
        #record the different fitting runs into structure 
        RecBu[s] = pt.softmax(rbm3.bu,0)
        RecLp[s] = pt.softmax(indepAr.logpi,0)
        RecUtrue[s] = ar.expand_mn(Utrue,K).mean(dim=0)
        RecEmLog[s] = pt.softmax(emloglik_train,1).mean(dim=0)

    fig = plt.figure(figsize=(8,8))
    plt.subplot(3,1,1)
    sb.lineplot(data=TT[(TT.iter>0) & (TT.type=='test')]
            ,y='crit',x='iter',hue='model')
    plt.ylabel('Test coserr')
    plt.subplot(3,1,2)
    sb.lineplot(data=TT[(TT.iter>0) & (TT.type=='compl')]
            ,y='crit',x='iter',hue='model')
    plt.ylabel('Compl coserr')
    plt.subplot(3,1,3)
    plt.plot(HH.T)
    plt.ylabel('Theta')

    # Get the final error and the true pott models
    plot_evaluation(DD)

    # OPtional: Plot the last maps of prior estimates
    # plot_Uhat_maps([None,indepAr,rbm3,Mtrue.arrange],emloglik_test[0:1],grid)
    # Optimal: plot the pmaps
    plot_P_maps([RecEmLog.mean(dim=0),
                RecLp.mean(dim=0),
                RecBu.mean(dim=0),
                pt.softmax(rbm.bu,0)],grid)

    plot_individual_Uhat(Models,Utrue[0:1],emloglik_train[0:1],
                    grid,style='mixed')
    # plot_individual_Uhat(Models,Utrue[0:1],emloglik_train[0:1],
    #                grid,style='argmax')
    pass

def simulation_chain():
    K = 3
    P = 5
    num_subj=100
    batch_size=100
    n_epoch=100
    logpi = 2.5
    num_sim = 10
    theta = 1.3
    # Multinomial 
    w = 1.5
    # MixGaussian 
    sigma2 = 0.5
    N = 10 

    eneg_iter = 6
    epos_iter = 6

    pt.set_default_dtype(pt.float32)
    TT=pd.DataFrame()
    DD=pd.DataFrame()
    HH = np.zeros((num_sim,n_epoch))

    # REcorded bias parameter 
    RecBu = pt.zeros((num_sim,K,P))
    RecLp = pt.zeros((num_sim,K,P))
    RecUtrue = pt.zeros((num_sim,K,P))
    RecEmLog = pt.zeros((num_sim,K,P))

    # Make a new emission model for the simulation
    emissionM = em.MultiNomial(K=K, P=P)
    emissionM.w = pt.tensor(w)
    # emissionM = em.MixGaussian(K,N,P)
    # emissionM.sigma2 = pt.tensor(sigma2)

    for s in range(num_sim):

        # Make the data
        Ytrain,Ytest,Utrue,Mtrue,grid = make_cmpRBM_chain(P,K,
            num_subj=num_subj,theta_w=theta,emission_model=emissionM,logpi=logpi)
        emloglik_train = Mtrue.emission.Estep(Y=Ytrain)
        emloglik_test = Mtrue.emission.Estep(Y=Ytest)

        # Generate partitions for region-completion testing
        part = pt.arange(0,5)

        # Independent spatial arrangement model
        indepAr = ar.ArrangeIndependent(K=K,P=P,spatial_specific=True,remove_redundancy=False)
        indepAr.name='idenp'
        indepAr,T,theta1 = train_sml(indepAr,Mtrue.emission,Ytrain,Ytest,
                part=part,n_epoch=n_epoch,batch_size=num_subj)

        # Gte the true arrangement model 
        rbm = Mtrue.arrange
        rbm.name = 'true'

        # Covolutional
        rbm3 = ar.cmpRBM(K,P,Wc=rbm.Wc,
                            theta=0.3,
                            eneg_iter=eneg_iter,
                            epos_iter=epos_iter,
                            eneg_numchains=num_subj)
        rbm3.bu = pt.zeros((K,P))
        rbm3.bu = indepAr.logpi.detach().clone()
        rbm3.bu = rbm.bu.detach().clone()
        rbm3.name=f'cRBM_Wc'
        rbm3.fit_W = True
        rbm3.fit_bu = True
        rbm3.alpha = 1

        # Make list of candidate models
        Models = [indepAr,rbm3,rbm]

        TH = [theta1]
        for m in Models[1:2]:

            m, T1,theta_hist = train_sml(m,Mtrue.emission,Ytrain,Ytest,part,
                batch_size=batch_size,
                n_epoch=n_epoch)
            TH.append(theta_hist)
            T = pd.concat([T,T1],ignore_index=True)

        # Evaluate overall
        D = eval_arrange(Models,Mtrue.emission,Ytrain,Ytest,Utrue=Utrue)
        D1 = eval_arrange_compl(Models,Mtrue.emission,Ytest,part=part,Utrue=Utrue)

        DD = pd.concat([DD,D,D1],ignore_index=True)
        TT = pd.concat([TT,T],ignore_index=True)
        HH[s,:]= TH[1][rbm3.get_param_indices('theta'),:]
        
        #record the different fitting runs into structure 
        RecBu[s] = pt.softmax(rbm3.bu,0)
        RecLp[s] = pt.softmax(indepAr.logpi,0)
        RecUtrue[s] = ar.expand_mn(Utrue,K).mean(dim=0)
        RecEmLog[s] = pt.softmax(emloglik_train,1).mean(dim=0)

    # Plot all the expectations over the 5 nodes
    plt.figure()
    plt.subplot(3,2,1)
    plt.plot(pt.softmax(rbm.bu,0).t())
    plt.ylim([0,1])
    plt.title('Bias term')

    plt.subplot(3,2,2)
    U = ar.expand_mn(Utrue,3)
    plt.plot(pt.mean(RecUtrue,0).t())
    plt.ylim([0,1])
    plt.title('True maps')

    plt.subplot(3,2,3)
    plt.plot(pt.mean(RecEmLog,0).t())
    plt.ylim([0,1])
    plt.title('Evidence')

    plt.subplot(3,2,4)
    plt.plot(pt.mean(RecLp,0).t())
    plt.ylim([0,1])
    plt.title('Independent Arrange')

    plt.subplot(3,2,5)
    plt.plot(pt.mean(RecBu,0).t())
    plt.ylim([0,1])
    plt.title('RBM3')


    fig = plt.figure(figsize=(8,8))
    plt.subplot(3,1,1)
    sb.lineplot(data=TT[(TT.iter>0) & (TT.type=='test')]
            ,y='crit',x='iter',hue='model')
    plt.ylabel('Test coserr')
    plt.subplot(3,1,2)
    sb.lineplot(data=TT[(TT.iter>0) & (TT.type=='compl')]
            ,y='crit',x='iter',hue='model')
    plt.ylabel('Compl coserr')
    plt.subplot(3,1,3)
    plt.plot(HH.T)
    plt.ylabel('Theta')

    plot_evaluation(DD)
    pass



def test_cmpRBM_Estep():
    K =5
    N = 20
    num_subj=500
    sigma2=0.5
    batch_size=20
    n_epoch=30
    pt.set_default_dtype(pt.float32)

    Ytrain,Ytest,Utrue,Mtrue,grid = make_cmpRBM_data(10,K,N=N,
        num_subj=num_subj,
        theta_mu=20,theta_w=1.0,sigma2=sigma2,
        do_plot=0)
        # Ytrain,Ytest,Utrue,Mtrue,grid = make_mrf_data(10,K,N=N,
        #         num_subj=num_subj,
        #         theta_mu=20,theta_w=2,sigma2=sigma2,
        #         do_plot=1)

    emloglik_train = Mtrue.emission.Estep(Y=Ytrain)
    emloglik_test = Mtrue.emission.Estep(Y=Ytest)
    P = Mtrue.emission.P
    M = Mtrue.arrange

    Uhat = pt.softmax(emloglik_train + M.bu,dim=1) # Start with hidden = 0
    for i in range(5):
        wv = pt.matmul(Uhat,M.W.t())
        Eh = pt.softmax(wv,1)
        wh = pt.matmul(Eh, M.W)
        grid.plot_maps(Uhat[0],cmap='jet',vmax=1,grid=(6,5),offset = i*5+1)
        Uhat = pt.softmax(wh + M.bu + emloglik_train,1)


    # D = eval_arrange([Mtrue.arrange],emloglik_train,emloglik_test,Utrue)
    pass


if __name__ == '__main__':
    # compare_gibbs()
    # train_rbm_to_mrf2('notebooks/sim_500.pt',n_hidden=[30,100],batch_size=20,n_epoch=20,sigma2=0.5)
    # simulation_2()
    simulation_chain()
    # simulation_2() 
    # plot_evaluation2()
    pass
    # test_cmpRBM_Estep()
    # test_sample_multinomial()
    # train_RBM()
