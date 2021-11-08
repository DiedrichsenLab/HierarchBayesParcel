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

def dict_learn_rep(Y,K=5,num=1):
    num_subj,P,N = Y.shape
    Vsubj = np.empty((num_subj,K,N))
    Vhat = np.empty((num,K,N))
    iter = np.empty((num,))
    loss = np.empty((num,))
    for s in range(num_subj):
        print(f"subj:{s}")
        Vhat = np.empty((num,K,N))
        for i in range(num):
            # Determine random starting value
            V_init = random_V(K,N)
            U_init = np.random.uniform(0,1,(P,N))
            Uhat,Vhat[i,:,:],errors,iter[i] = dict_learning(Y[s,:,:],alpha = 0.1, n_components=K,
            method='cd',positive_code=True,
            code_init=U_init, dict_init=V_init,
            return_n_iter=True,max_iter=200)
            loss[i] = errors[-1]
        # Sort the solutions by the loss
        i=loss.argmin()
        Vsubj[s,:,:] = Vhat[i,:,:]
    return Vsubj

def vmatch(V1,V2):
    """Gets the mean minimal distances of every vector in V1
        to any vector in V2.
    """
    num_sub,K1,N = V1.shape
    num_sub,K2,N = V2.shape
    M = np.zeros((num_sub,num_sub)) # Mean correspondence
    for i in range(num_sub):
        for j in range(num_sub):
            if i==j:
                M[i,j]=np.nan
            else:
                M[i,j]=(V1[i,:,:] @ V2[j,:,:].T).max(axis=1).mean()
    return np.nanmean(M,axis=1), M


def vmatch_baseline(K=[5,5],N=30,num=20):
    V = np.empty((2,),'object')
    for set in range(2):
        V[set]=np.empty((num,K[set],N))
        for i in range(num):
            V[set][i,:,:]=random_V(K[set],N)
    vm,M = vmatch(V[0],V[1])
    return np.mean(vm)

def vmatch_baseline_fK():
    kmax= 30
    vm = np.empty((kmax,))
    kk = np.arange(1,kmax+1)
    for k in kk:
        vm[k-1] = vmatch_baseline([k,3],60)
    plt.plot(kk,vm)
    pass

def correspondence_sim(K=[3,8],N=42,P=100,sig=[2,2]):
    # Checks on the power of correspondence test between
    num=1
    num_subj=3

    # Generate data
    V = np.empty((2,),'object')
    U = np.empty((2,),'object')
    Y = np.empty((2,),'object')
    for set in range(2):
        V[set] = random_V(K[set],N)
        U[set] = np.empty((num_subj,P,K[set]))
        Y[set] = np.empty((num_subj,P,N))
        for s in range(num_subj):
            U[set][s,:,:] = random_U(P,K[set])
            Y[set][s,:,:] = random_Y(U[set][s,:,:],V[set],sig[set])

    # Estimate V repeatedly
    Vhat=np.empty((2,),'object')
    for set in range(2):
        Vhat[set]=dict_learn_rep(Y[set],K=K[set])

    # Establish match within and across sets
    match = [[0,0],[0,1],[1,1],[1,0]]
    M = np.zeros((4,num_subj))
    for l,m in enumerate(match):
        M[l,:],mV = vmatch(Vhat[m[0]],Vhat[m[1]])
    pass


if __name__ == '__main__':
    vmatch_baseline_fK()
    # correspondence_sim()