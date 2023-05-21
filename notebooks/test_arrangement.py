import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import matplotlib.pyplot as plt
import torch as pt
import torchvision.transforms as transforms
import arrangements as ar
import emissions as em
import full_model as fm
import spatial as sp
import evaluation as ev
import pandas as pd
import seaborn as sb
import copy

from FusionModel.evaluate import calc_test_dcbc
from generativeMRF.depreciated import FullModel

# pytorch cuda global flag
# pt.cuda.is_available = lambda : False
pt.set_default_tensor_type(pt.cuda.FloatTensor
                           if pt.cuda.is_available() else
                           pt.FloatTensor)

def gaussian_kernel(size, sigma=1):
    """Generate a Gaussian kernel of the given size and standard deviation"""
    size = int(size) // 2
    coords = pt.meshgrid(*[pt.arange(-size, size+1)]*N)
    distances_sq = sum([(x**2) for x in coords])
    g = torch.exp(-distances_sq / (2*sigma**2))
    return g / g.sum()

def gaussian_smoothing_Nd(image):
    """Apply a 3x3 Gaussian kernel smoothing on an Nd image"""
    kernel = gaussian_kernel(3, sigma=1).unsqueeze(0).unsqueeze(0).to(image.device)
    smoothed = F.conv2d(image.unsqueeze(0), kernel.repeat(image.shape[1], 1, 1, 1), padding=1)
    return smoothed.squeeze(0)

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
    MT = FullModel(arrangeT,emissionT)

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

def make_potts_data(width=30, K=5, N=20, nsubj=10, sigma2=0.2, theta_mu=120,
                    theta_w=1.5, inits=None):
    """Making a full model contains an arrangement model and one or more
       emission models with desired settings
    Args:
        K: the number of clusers
        P: the number of voxels
        nsubj_list: the number of subjects in emission models
        M: the number of conditions per emission model
        num_part: the number of partitions per emission model
        common_kappa: if True, using common kappa. Otherwise, separate kappas
        same_subj: if True, the same set of subjects across emission models
    Returns:
        M: the full model object
    """
    # Step 1: Create the true model
    grid = sp.SpatialGrid(width=width, height=width)
    arrangeT = ar.PottsModel(grid.W, K=K, remove_redundancy=False)
    arrangeT.name = 'Potts'
    arrangeT.random_smooth_pi(grid.Dist, theta_mu=theta_mu, centroids=inits)
    arrangeT.theta_w = pt.tensor(theta_w)

    # Step 2: create the emission model and sample from it with a specific signal
    emissionT = em.MixGaussian(K, N, width*width)
    emissionT.num_subj = nsubj
    emissionT.sigma2 = pt.tensor(sigma2)

    # Step 3: Create the full model
    T = fm.FullMultiModel(arrangeT,[emissionT])
    T.initialize()

    # Sampling individual Us and data, data_test
    U, Y_train = T.sample()
    Y_test = []
    for m, Us in enumerate(T.distribute_evidence(U)):
        Y_test.append(T.emissions[m].sample(Us))

    return Y_train[0], Y_test[0], U, T, grid

def make_cmpRBM_data(width=10, K=5, N=10,num_subj=20, theta_mu=20,
                     theta_w=1.0, emission_model=None, do_plot=1):
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

    MT = FullModel(arrangeT,emission_model)

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
        plt.figure(figsize=(20,10))

    for i in range (10):
        ph, H = arrangeT.sample_h(U)
        pu, U = arrangeT.sample_U(H)
        if do_plot>1:
            u = ar.compress_mn(U)
            grid.plot_maps(u[8],cmap='tab10',vmax=K,grid=[2,5],offset=i+1)

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


    MT = FullModel(arrangeT,emission_model)
    # Step 4: Generate data by sampling from the above model
    Utrue = MT.arrange.sample(num_subj,50)
    Ytrain = MT.emission.sample(Utrue)
    Ytest = MT.emission.sample(Utrue)  # Testing data

    return Ytrain, Ytest, Utrue, MT , grid

def make_train_model(model_name='cmpRBM', K=3, P=5, num_subj=20, eneg_iter=10,
                     epos_iter=10, Wc=None, theta=None, fit_W=True, fit_bu=False, lr=1):
    if model_name.startswith('idenp'):
        # 1 - Independent spatial arrangement model
        M = ar.ArrangeIndependent(K=K, P=P, spatial_specific=True,
                                  remove_redundancy=False)
        M.random_params()
        M.name = model_name
    elif model_name == 'cRBM_W':
        # Boltzmann with a arbitrary fully connected model - P hiden nodes
        n_hidden = P
        M = ar.cmpRBM(K, P, nh=n_hidden, eneg_iter=eneg_iter,
                      epos_iter=epos_iter, eneg_numchains=num_subj)
        M.name=f'cRBM_{n_hidden}'
        M.W = pt.randn(n_hidden,P) * 0.1
        M.alpha = lr
        M.fit_W = fit_W
        M.fit_bu = fit_bu
    elif model_name == 'cRBM_Wc':
        # Covolutional Boltzman machine with the true neighbourhood matrix
        # theta_w in this case is not fit.
        M = ar.cmpRBM(K, P, Wc=Wc, theta=theta, eneg_iter=eneg_iter,
                      epos_iter=epos_iter, eneg_numchains=num_subj)
        M.name = 'cRBM_Wc'
        M.fit_W = fit_W
        M.fit_bu = fit_bu
        M.alpha = lr
    else:
        raise ValueError('Unknown model name')

    return M

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
    theta_hist = pt.zeros((arM.nparams,n_epoch))
    CE = pt.zeros((n_epoch,))
    # Intialize negative sampling
    for epoch in range(n_epoch):
        # Get test error
        EU, _ = arM.Estep(emlog_train, gather_ss=False)
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

        # Record the cross entropy parameters
        if arM.name.startswith('idenp'):
            CE[epoch] = 0
        else:
            CE[epoch] = ev.cross_entropy(pt.softmax(emlog_train, dim=1),
                                         arM.eneg_U)
            # CE[epoch] = pt.abs(pt.softmax(emlog_train, dim=1) - arM.eneg_U).sum()

        theta_hist[:,epoch]=arM.get_params()

    # Make a data frame for the results
    T=pd.DataFrame()
    for i, ct in enumerate(crit_types):
        T1 = pd.DataFrame({'model':[arM.name]*n_epoch,
                        'type':[ct]*n_epoch,
                        'iter':np.arange(n_epoch),
                        'crit':CR[i]})
        T = pd.concat([T,T1],ignore_index=True)

    return arM, T, theta_hist, CE

def eval_dcbc(models, emM, Ytrain, Ytest, grid, Utrue_group, Utrue_indiv,
              max_dist=10, bin_width=1):
    D= pd.DataFrame()
    emloglik_train = emM.Estep(Ytrain)
    group_par, indiv_par = [], []

    for m in models:
        smooth = 0
        if m=='data':
            this_Ugroup = pt.softmax(emloglik_train.sum(dim=0), dim=0).argmax(dim=0)
            this_Uindiv = pt.softmax(emloglik_train,1).argmax(dim=1)
            name = m
            model_type = 'data'
        elif m=='Utrue':
            this_Ugroup = Utrue_group
            this_Uindiv = Utrue_indiv
            name = m
            model_type = 'true'
        else:
            # EU,_ = m.Estep(emloglik_train, gather_ss=False)
            if m.name.startswith('idenp'):
                this_Ugroup = m.marginal_prob().argmax(dim=0)
                this_Uindiv = m.estep_Uhat.argmax(dim=1)
                smooth = float(m.name.split('_')[1])
                model_type = 'idenp'
            elif m.name.startswith('cRBM'):
                this_Ugroup = pt.softmax(m.bu, dim=0).argmax(dim=0)
                this_Uindiv = m.epos_Uhat.argmax(dim=1)
                model_type = 'cRBM'
            else:
                raise NameError('Unknown model name')
            name = m.name

        dcbc_group = calc_test_dcbc(this_Ugroup, Ytest, grid.Dist,
                                    max_dist=int(max_dist), bin_width=bin_width)
        dcbc_indiv = calc_test_dcbc(this_Uindiv, Ytest, grid.Dist,
                                    max_dist=int(max_dist), bin_width=bin_width)

        group_par.append(this_Ugroup)
        indiv_par.append(this_Uindiv)

        dict = {'model':[name],
                'type':['test'],
                'smooth': smooth,
                'arrangement': model_type,
                'dcbc_group':dcbc_group.mean().item(),
                'dcbc_indiv':dcbc_indiv.mean().item()}
        D = pd.concat([D,pd.DataFrame(dict)],ignore_index=True)

    return D, group_par, indiv_par

def eval_arrange(models,emM,Ytrain,Ytest,Utrue):
    D= pd.DataFrame()
    Utrue_mn = ar.expand_mn(Utrue,emM.K)
    emloglik_train = emM.Estep(Ytrain)
    
    for m in models:
        smooth = 0
        if m=='data':
            EU = pt.softmax(emloglik_train,1)
            name = m
            model_type = 'data'
        elif m=='Utrue':
            EU = Utrue_mn
            name = m
            model_type = 'true'
        else: 
            # EU,_ = m.Estep(emloglik_train, gather_ss=False)
            if m.name.startswith('idenp'):
                EU = m.estep_Uhat
                smooth = float(m.name.split('_')[1])
                model_type = 'idenp'
            elif m.name.startswith('cRBM'):
                EU = m.epos_Uhat
                model_type = 'cRBM'
            else:
                raise NameError('Unknown model name')
            name = m.name
        uerr_test1= ev.u_abserr(Utrue_mn,EU)
        cos_err= ev.coserr(Ytest,emM.V,EU,adjusted=False,
                 soft_assign=False).mean(dim=0).item()
        Ecos_err= ev.coserr(Ytest,emM.V,EU,adjusted=False,
                 soft_assign=True).mean(dim=0).item()

        dict ={'model':[name],
               'type':['test'],
               'smooth':smooth,
               'arrangement': model_type,
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
    n_models = len(pmaps)
    K = pmaps[0].shape[0]

    plt.figure(figsize=(K*3, n_models*3))
    for i,m in enumerate(pmaps):
        grid.plot_maps(m,cmap='jet',vmax=1,grid=(n_models,K),offset=i*K+1)

    plt.show()

def plot_U_maps(pmaps, grid, title):
    n_models = len(pmaps)

    plt.figure(figsize=(n_models * 3, 4))
    for i,m in enumerate(pmaps):
        grid.plot_maps(m,cmap='tab20',vmax=19,grid=(1, n_models),offset=i+1)
        plt.title(title[i])

    plt.show()

def plot_individual_Uhat(models,Utrue, emloglik,grid,style='prob'):
    # Get the expectation
    n_models = len(models)+2 
    K = emloglik.shape[1]
    P = emloglik.shape[2]
    
    Uh = []
    height = 2 if style=='mixed' else 1
    plt.figure(figsize=(n_models*3, height*4))

    # Uh order: data -> models -> Utrue
    Uh.append(pt.softmax(emloglik[0:1],dim=1))
    for i,m in enumerate(models):
        A,_=m.Estep(emloglik[0:1])
        Uh.append(A)
    Uh.append(ar.expand_mn(Utrue[0:1], K))

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

    plt.show()

def plot_evaluation(D, criteria=['uerr','cos_err','Ecos_err','dcbc_group','dcbc_indiv'],
                    types=['test','compl']):
    # Get the final error and the true pott models
    ncrit = len(criteria)
    ntypes = len(types)
    plt.figure(figsize=(5*ncrit, 5*ntypes))
    for j in range(ntypes): 
        for i in range(ncrit): 
            plt.subplot(ntypes,ncrit,i+j*ncrit+1)
            # sb.barplot(data=D[D.type==types[j]], x='model', y=criteria[i])

            df = D[(D.type == types[j]) & (D.arrangement == 'idenp')]
            sb.lineplot(data=df, x='smooth', y=criteria[i],
                        err_style="bars", markers=False)

            emlog = D[(D.type == types[j]) & (D.arrangement == 'data')]
            plt.axhline(emlog[criteria[i]].mean().item(), color='k', ls=':',
                        label='data')

            rbm_wc = D[(D.type == types[j]) & (D.model == 'cRBM_Wc')]
            plt.axhline(rbm_wc[criteria[i]].mean().item(), color='r', ls=':',
                        label='cRBM_Wc')

            rbm_wc = D[(D.type == types[j]) & (D.model == 'Utrue')]
            plt.axhline(rbm_wc[criteria[i]].mean().item(), color='b', ls=':',
                        label='Utrue')

            plt.title(f'{criteria[i]}{types[j]}')
            plt.legend()
            # plt.xticks(rotation=45)

    plt.suptitle(f'final errors')
    plt.tight_layout()
    plt.show()

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

def simulation_2(K=5, width=50, num_subj=20, batch_size=20, n_epoch=120, theta=1.2,
                 theta_mu=180, emission='gmm', epos_iter=20, eneg_iter=20, num_sim=10):
    P = width * width
    if emission == 'gmm': # MixGaussian
        sigma2 = 0.2
        N = 10
        emissionM = em.MixGaussian(K, N, P)
        emissionM.sigma2 = pt.tensor(sigma2)
    elif emission == 'mn': # Multinomial
        w = 2.0
        emissionM = em.MultiNomial(K=K, P=P)
        emissionM.w = pt.tensor(w)
    
    # Record the results
    TT=pd.DataFrame()
    DD=pd.DataFrame()
    HH = pt.zeros((num_sim,n_epoch))
    CE_rbm1 = pt.zeros((num_sim, n_epoch))
    CE_rbm2 = pt.zeros((num_sim, n_epoch))
    GM, IM = [], []

    # REcorded bias parameter
    SD = np.concatenate((np.linspace(0.1,1,2), np.linspace(1.5,3,2)))
    SD = np.round(SD, decimals=2)
    Rec = pt.zeros((len(SD)+4, num_sim, K, P)) # unsmooth + 2 rbms + 1 emloglik

    # Generate partitions for region-completion testing
    num_part = 4
    p = pt.ones(num_part) / num_part
    part = pt.multinomial(p, P, replacement=True)
    for s in range(num_sim):
        Ytrain,Ytest,Utrue,Mtrue,grid = make_cmpRBM_data(width,K,N=N,
                                        num_subj=num_subj, theta_mu=theta_mu,
                                        theta_w=theta, emission_model=emissionM,
                                        do_plot=0)

        # Get the smoothed training data
        Ytrain_smooth = []
        for smooth in SD:
            blur_transform = transforms.GaussianBlur(kernel_size=5, sigma=smooth)
            Ys = blur_transform(Ytrain.view(Ytrain.shape[0],-1,width,width))
            Ys = Ys.view(Ytrain.shape[0],Ytrain.shape[1],-1)
            Ytrain_smooth.append(Ys)

        emloglik_train = Mtrue.emission.Estep(Ytrain)
        emloglik_test = Mtrue.emission.Estep(Ytest)
        P = Mtrue.emission.P

        # Get the true arrangement model and its loglik
        rbm = Mtrue.arrange
        rbm.name = 'true'

        # Make list of fitting models
        Models, fitted_M = [], []
        fitting_names = ['idenp_0'] + [f'idenp_{s}' for s in SD] + ['cRBM_Wc','cRBM_W']
        Y_fit = [Ytrain] + Ytrain_smooth + [Ytrain, Ytrain]
        for nam in fitting_names:
            Models.append(make_train_model(model_name=nam, K=K, P=P,
                                           num_subj=num_subj, eneg_iter=eneg_iter,
                                           epos_iter=epos_iter, Wc=rbm.Wc, theta=None,
                                           fit_W=True, fit_bu=False, lr=0.1))

        # Train different arrangement model
        TH, CE = [], []
        T = pd.DataFrame()
        for i, m in enumerate(Models):
            # Give the model the true bias/W for rbms
            if m.name.startswith('cRBM'):
                # m.W = rbm.W.detach().clone()
                m.bu = rbm.bu.detach().clone()

            m, T1, theta_hist, ce = train_sml(m, Mtrue.emission, Y_fit[i],
                                              Ytest, part, batch_size=batch_size,
                                              n_epoch=n_epoch)
            fitted_M.append(m)
            TH.append(theta_hist)
            CE.append(ce)
            T = pd.concat([T,T1],ignore_index=True)

        # Evaluate overall
        # 1. u_absolute error, cos_err, and expected cos_err
        D = eval_arrange(['data'] + fitted_M + ['Utrue'],
                         Mtrue.emission, Ytrain, Ytest, Utrue=Utrue)

        # 2. DCBC
        binWidth = 5
        max_dist = binWidth * pt.ceil(grid.Dist.max() / binWidth)
        D1, group_map, indiv_map = eval_dcbc(['data'] + fitted_M + ['Utrue'], Mtrue.emission,
                                             Ytrain, Ytest, grid,
                                             pt.softmax(rbm.bu, dim=0).argmax(dim=0), Utrue,
                                             max_dist=max_dist, bin_width=binWidth)

        GM.append(group_map)
        IM.append(indiv_map)
        # 3. Region completion test
        # D1 = eval_arrange_compl(fitted_M, Mtrue.emission, Ytest,
        #                         part=part, Utrue=Utrue)
        res = pd.merge(D, D1, how='outer')
        DD = pd.concat([DD, res],ignore_index=True)
        TT = pd.concat([TT, T],ignore_index=True)

        # Record the theta for rbm_Wc model only
        HH[s,:]= TH[-2][fitted_M[-2].get_param_indices('theta'),:]

        # Record cross entropy for rbms
        CE_rbm1[s, :] = CE[-2]
        CE_rbm2[s, :] = CE[-1]
        
        # record the different fitting runs into structure
        Rec[0,s,:,:] = pt.softmax(emloglik_train, 1).mean(dim=0) # first is data
        for j, fm in enumerate(fitted_M):
            if fm.name.startswith('idenp'):
                Rec[j+1,s,:,:] =  pt.softmax(fm.logpi, 0)
            elif fm.name.startswith('cRBM'):
                Rec[j+1,s,:,:] = pt.softmax(fm.bu, 0)
            else:
                raise ValueError('Unknown model name')
        # Rec[-1,s,:,:] = ar.expand_mn(Utrue, K).mean(dim=0)

    # Plot learning curves by epoch
    fig = plt.figure(figsize=(10,10))
    plt.subplot(2, 2, 1)
    plt.plot(CE_rbm1.T.cpu().numpy(), linestyle='-', label='rbm_Wc')
    plt.plot(CE_rbm2.T.cpu().numpy(), linestyle=':', label='rbm_W')
    plt.ylabel('Cross Entropy')
    plt.legend(['rbm_Wc (solid)','rbm_W (dotted)'])
    plt.subplot(2, 2, 2)
    sb.lineplot(data=TT[(TT.iter>0) & (TT.type=='test')], y='crit',
                x='iter', hue='model')
    plt.ylabel('Test coserr')
    plt.subplot(2, 2, 3)
    sb.lineplot(data=TT[(TT.iter>0) & (TT.type=='compl')]
            ,y='crit',x='iter',hue='model')
    plt.ylabel('Compl coserr')
    plt.subplot(2, 2, 4)
    plt.plot(HH.T.cpu().numpy())
    plt.axhline(y=HH[:,-1].cpu().numpy().mean(), color='r', linestyle='-')
    plt.axhline(y=theta, color='k', linestyle='-')
    plt.ylabel('Theta')
    plt.show()

    # records = [RecEmLog, RecLp1, RecLp2, RecLp3, RecLp4, RecLp5, RecBu1, RecBu2]
    return grid, DD, Rec, rbm, fitted_M, Utrue, emloglik_train, GM, IM

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
    # simulation_chain()
    grid, DD, records, rbm, Models, Utrue, emloglik_train, GM, IM = simulation_2(theta_mu=240,
                                                                                 num_sim=20)

    # Get the final error and the true pott models
    plot_evaluation(DD, types=['test'])

    # OPtional: Plot the last maps of prior estimates
    # plot_Uhat_maps([None,indepAr,rbm3,Mtrue.arrange],emloglik_test[0:1],grid)

    # Optimal: plot the prob maps by K
    plot_P_maps(pt.cat((records.mean(dim=1), pt.softmax(rbm.bu, 0).unsqueeze(0)), dim=0),
                grid)

    # plot the group reconstructed U maps
    plot_U_maps(pt.stack(GM[0]), grid, title=['data'] + [m.name for m in Models] + ['true'])

    plot_individual_Uhat(Models, Utrue[0:1], emloglik_train[0:1],
                         grid, style='mixed')
    # plot_individual_Uhat(Models,Utrue[0:1],emloglik_train[0:1],
    #                grid,style='argmax')
    pass


    # plot_evaluation2()
    # test_cmpRBM_Estep()
    # test_sample_multinomial()
    # train_RBM()
