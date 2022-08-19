import os  # to handle path information
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp,log,sqrt
import torch as pt
import arrangements as ar
from arrangements import sample_multinomial, compress_mn, expand_mn
import emissions as em
import full_model as fm
import spatial as sp
import copy
import evaluation as ev
import pandas as pd
import seaborn as sb
import time
from test_arrangement import eval_arrange, eval_arrange_compl, train_sml

"""_summary_
This script check with a fully observed multinomial RDM. 
"""


class mRBM(ar.ArrangementModel):
    def __init__(self, K, P, nh=None, Wc = None, theta=None, eneg_iter=3,eneg_numchains=77):
        """convolutional multinomial (categorial) restricted Boltzman machine
        for learning of brain parcellations for probabilistic input
        Uses variational stochastic maximum likelihood for learning

        Args:
            K (int): number of classes
            P (int): number of brain locations
            nh (int): number of hidden multinomial nodes
            Wc (tensor): 2d/3d-tensor for connecticity weights
            theta (tensor): 1d vector of parameters
            eneg_iter (int): HOw many iterations for each negative step. Defaults to 3.
            eneg_numchains (int): How many chains. Defaults to 77.
        """
        self.K = K
        self.P = P
        self.Wc  = Wc
        self.bu = pt.randn(K,P)
        if Wc is None:
            if nh is None:
                raise(NameError('Provide Connectivty kernel (Wc) matrix or number of hidden nodes (nh)'))
            self.nh = nh
            self.W = pt.randn(nh,P)
            self.theta = None
            self.set_param_list(['bu','W'])
        else:
            if Wc.ndim==2:
                self.Wc= Wc.view(Wc.shape[0],Wc.shape[1],1)
            self.nh = Wc.shape[0]
            if theta is None:
                self.theta = pt.randn((self.Wc.shape[2],))
            else:
                self.theta = pt.tensor(theta)
                if self.theta.ndim ==0:
                    self.theta = self.theta.view(1)
            self.W = (self.Wc * self.theta).sum(dim=2)
            self.set_param_list(['bu','theta'])
        self.gibbs_U = None # samples from the hidden layer for negative phase
        self.alpha = 0.01
        self.eneg_iter = eneg_iter
        self.eneg_numchains = eneg_numchains
        self.fit_bu = True
        self.fit_W = True


    def sample_h(self, U):
        """Sample hidden nodes given an activation state of the outer nodes
        Args:
            U (NxKxP tensor): Indicator or probability tensor of outer layer
        Returns:
            p_h: (N x nh tensor): probability of the hidden nodes
            sample_h (N x nh tensor): 0/1 values of discretely sampled hidde nodes
        """
        wv = pt.matmul(U,self.W.t())
        p_h = pt.softmax(wv,1)
        sample_h = sample_multinomial(p_h,kdim=1)
        return p_h, sample_h

    def sample_U(self, h, emloglik = None):
        """ Returns a sampled U as a unpacked indicator variable
        Args:
            h tensor: Hidden states (NxKxnh)
        Returns:
            p_u: Probability of each node [N,K,P] array
            sample_U: One-hot encoding of random sample [N,K,P] array
        """
        N = h.shape[0]
        act = pt.matmul(h, self.W) + self.bu
        if emloglik is not None:
            act += emloglik
        p_u = pt.softmax(act ,1)
        sample = sample_multinomial(p_u,kdim=1)
        return p_u, sample

    def sample(self,num_subj,iter=10):
        """Draw new subjects from the model

        Args:
            num_subj (int): Number of subjects
            iter (int): Number of iterations until burn in
        """
        p = pt.ones(self.K)
        u = pt.multinomial(p,num_subj*self.P,replacement=True)
        u = u.reshape(num_subj,self.P)
        U = expand_mn(u,self.K)
        for i in range (iter):
            _,h = self.sample_h(U)
            _,U = self.sample_U(h)
        u = compress_mn(U)
        return u

    def marginal_prob(self):
        # If not given, then initialize:
        if self.gibbs_U is None:
            return pt.softmax(self.bu,0)
        else:
            pi  = pt.mean(self.gibbs_U,dim=0)
        return pi

    def Estep(self, emloglik, gather_ss=True,iter=None):
        """ Positive Estep for the multinomial boltzman model
        Uses mean field approximation to posterior to U and hidden parameters.
        Parameters:
            emloglik (pt.tensor):
                emission log likelihood log p(Y|u,theta_E) a numsubj x K x P matrix
            gather_ss (bool):
                Gather Sufficient statistics for M-step (default = True)

        Returns:
            Uhat (pt.tensor):
                posterior p(U|Y) a numsubj x K x P matrix
            ll_A (pt.tensor):
                Nan - returned for consistency
        """
        if type(emloglik) is np.ndarray:
            emloglik=pt.tensor(emloglik,dtype=pt.get_default_dtype())
        N=emloglik.shape[0]
        U = pt.softmax(emloglik + self.bu,dim=1) # Start with hidden = 0
        wv = pt.matmul(U,self.W.t())
        Hhat = pt.softmax(wv,1)
        if gather_ss:
            self.epos_U = U
            self.epos_Hhat = Hhat
        return U, pt.nan

    def Eneg(self, iter=None, use_chains=None, emission_model=None):
        # If no iterations specified - use standard
        if iter is None:
            iter = self.eneg_iter
        # If no markov chain are initialized, start them off
        if (self.gibbs_U is None):
            p = pt.softmax(self.bu,0)
            self.gibbs_U = sample_multinomial(p,
                    shape=(self.eneg_numchains,self.K,self.P),
                    kdim=0,
                    compress=False)
        # Grab the current chains
        if use_chains is None:
            use_chains = pt.arange(self.eneg_numchains)

        U = self.gibbs_U[use_chains]
        U0 = U.detach().clone()
        for i in range(iter):
            _,H = self.sample_h(U)
            _,U = self.sample_U(H)
        self.eneg_H = H
        self.eneg_U = U
        # Persistent: Keep the new gibbs samples around
        self.gibbs_U[use_chains]=U
        return self.eneg_U,self.eneg_H

    def Mstep(self):
        """Performs gradient step on the parameters

        Args:
            alpha (float, optional): [description]. Defaults to 0.8.
        """
        N = self.epos_Hhat.shape[0]
        M = self.eneg_H.shape[0]
        # Update the connectivity
        if self.fit_W:
            gradW = pt.matmul(pt.transpose(self.epos_Hhat,1,2),self.epos_U).sum(dim=0)/N
            gradW -= pt.matmul(pt.transpose(self.eneg_H,1,2),self.eneg_U).sum(dim=0)/M
            # If we are dealing with component Wc:
            if self.Wc is not None:
                gradW = gradW.view(gradW.shape[0],gradW.shape[1],1)
                self.theta += self.alpha * pt.sum(gradW*self.Wc,dim=(0,1))
                self.W = (self.Wc * self.theta).sum(dim=2)
            else:
                self.W += self.alpha * gradW

        # Update the bias term
        if self.fit_bu:
            gradBU =   1 / N * pt.sum(self.epos_U,0)
            gradBU -=  1 / M * pt.sum(self.eneg_U,0)
            self.bu += self.alpha * 2 * gradBU


def simulation_chain_obs():
    """This does a chain simulation where the U are fully observed

    """
    K = 3
    N = 20
    P = 5
    num_subj=50
    batch_size=50
    n_epoch=100
    logpi = 3
    num_sim = 30
    theta = 1.3
    pt.set_default_dtype(pt.float32)

    # Make true chain model and sample data from it
    grid = sp.SpatialChain(P=5)
    W = grid.get_neighbour_connectivity()
    W += pt.eye(W.shape[0])

    # Step 2: Initialize the parameters of the true model: Only ends are fixed
    Mtrue = mRBM(K,grid.P,Wc=W,theta=theta)
    Mtrue.name = 'mRDM'
    Mtrue.bu = pt.zeros((K,P))
    Mtrue.bu[0,0]=logpi
    Mtrue.bu[-1,-1]=logpi
    P = Mtrue.P

    TT=pd.DataFrame()
    DD=pd.DataFrame()
    HH = np.zeros((num_sim,n_epoch))
    BU = pt.zeros((num_sim,K,P))
    TH = []

    Utrue = pt.empty((num_sim,num_subj,K,P))
    for s in range(num_sim):

        # Make the data
        U = Mtrue.sample(num_subj,50)
        Utrue[s] = expand_mn(U,K)
        emloglik_train=Utrue[s].detach().clone()
        emloglik_train[emloglik_train==1] = 1e10
        emloglik_test=emloglik_train

        # Generate partitions for region-completion testing
        part = pt.arange(0,P)

        rbm = mRBM(K,P,Wc=Mtrue.Wc,
                            theta=0.5,
                            eneg_iter=10,
                            eneg_numchains=num_subj)
        rbm.bu = pt.zeros((K,P))
        # rbm3.bu = indepAr.logpi.detach().clone()
        # rbm3.bu = rbm.bu.detach().clone()
        rbm.name=f'fRBM'
        rbm.fit_W = True
        rbm.fit_bu = True
        rbm.alpha = 0.1

        # Make list of candidate models
        Models = [rbm,Mtrue]

        rbm, T1,theta_hist = train_sml(rbm,emloglik_train,emloglik_train, 
                part=part,
                batch_size=batch_size,
                n_epoch=n_epoch)

        # Evaluate overall
        D = eval_arrange(Models,emloglik_train,emloglik_test,U)
        D1 = eval_arrange_compl(Models,emloglik_test,part=part,Utrue=U)

        DD = pd.concat([DD,D,D1],ignore_index=True)
        TT = pd.concat([TT,T1],ignore_index=True)
        HH[s,:]= theta_hist[rbm.get_param_indices('theta'),:]
        BU[s] = rbm.bu.detach().clone()

    # Plot all the expectations over the 5 nodes
    plt.figure()
    plt.subplot(3,2,1)
    plt.plot(pt.softmax(Mtrue.bu,0).t())
    plt.ylim([0,1])
    plt.title('Bias term')

    plt.subplot(3,2,2)
    plt.plot(Utrue.mean(dim=(0,1)).t())
    plt.ylim([0,1])
    plt.title('True maps')

    plt.subplot(3,2,3)
    plt.plot(pt.softmax(emloglik_test,1).mean(dim=0).t())
    plt.ylim([0,1])
    plt.title('Evidence')

    plt.subplot(3,2,5)
    plt.plot(pt.mean(pt.softmax(BU,1),0).t())
    plt.ylim([0,1])
    plt.title('RBM3')


    fig = plt.figure(figsize=(8,8))
    plt.subplot(3,1,1)
    sb.lineplot(data=TT[(TT.iter>0) & (TT.type=='test')]
            ,y='crit',x='iter',hue='model')
    plt.ylabel('Test logpy')
    plt.subplot(3,1,2)
    sb.lineplot(data=TT[(TT.iter>0) & (TT.type=='compl')]
            ,y='crit',x='iter',hue='model')
    plt.ylabel('Compl logpy')
    plt.subplot(3,1,3)
    plt.plot(HH.T)
    plt.ylabel('Theta')

    # Get the final error and the true pott models
    plt.figure(figsize=(8,7))
    plt.subplot(2,2,1)
    sb.barplot(data=DD[DD.type=='test'],x='model',y='uerr')
    plt.title('uerr test')
    plt.subplot(2,2,2)
    sb.boxplot(data=DD[DD.type=='test'],x='model',y='logpy')
    plt.title('logpy test')
    plt.subplot(2,2,3)
    sb.barplot(data=DD[DD.type=='compl'],x='model',y='uerr')
    plt.title('uerr compl')
    plt.subplot(2,2,4)
    sb.boxplot(data=DD[DD.type=='compl'],x='model',y='logpy')
    plt.title('logpy compl')

    pass


if __name__ == '__main__':
    # compare_gibbs()
    # train_rbm_to_mrf2('notebooks/sim_500.pt',n_hidden=[30,100],batch_size=20,n_epoch=20,sigma2=0.5)
    simulation_chain_obs()
    # test_sample_multinomial()
    # train_RBM()
