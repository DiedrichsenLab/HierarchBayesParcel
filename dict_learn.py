from sklearn.decomposition import dict_learning, sparse_encode
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def random_V(K=5,N=20):
    V = np.random.normal(0,1,(K,N))
    V = V - V.mean(axis=1).reshape(-1,1)
    V = V / np.sqrt(np.sum(V**2,axis=1).reshape(-1,1))
    return V

def random_U(P=100,K=5,type='onegamma',alpha=2,beta=0.3):
    if type=='onegamma':
        u = np.random.choice(K,(P,))
        g = np.random.gamma(alpha,beta,(P,))
        U = np.zeros((P,K))
        for i in np.arange(K):
            U[u==i,i] = g[u==i]
    elif type=='iidgamma':
        U = np.random.gamma(alpha,beta,(P,K))
    elif type=='one':
        u = np.random.choice(K,(P,))
        U = np.zeros((P,K))
        for i in np.arange(K):
            U[u==i, i] = 1
    return U

def random_Y(U,V,eps=1):
    N = V.shape[1]
    P = U.shape[0]
    Y = U @ V + np.random.normal(0,eps/np.sqrt(N),(P,N))
    return Y

def check_consistency(P=100,K=5,N=40):
    num=10
    U = random_U(P,K,'onegamma')
    V = random_V(K,N)
    Y = random_Y(U,V,eps=1)
    Uhat = np.empty((num,P,K))
    Vhat = np.empty((num,K,N))
    iter = np.empty((num,))
    loss = np.empty((num,))

    for i in range(num):
        # Determine random starting value
        V_init = random_V(K,N)
        U_init = np.random.uniform(0,1,(P,N))
        Uhat[i,:,:],Vhat[i,:,:],errors = dict_learning (Y,alpha = 0.1, n_components=5,
            method='cd',positive_code=True,code_init=U_init, dict_init=V_init)
        loss[i] = errors[-1]
        iter[i] = len(errors)
    pass

if __name__ == '__main__':
    check_consistency()