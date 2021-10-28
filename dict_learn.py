from sklearn.decomposition import dict_learning, sparse_encode
import numpy as np
import pandas as pd
import seaborn as sns


# Dictonary learning: Toy example with V~normal, U ~ Gamma(2,0.3), Y = UV + eps 
K = 5 
N = 20 
P = 100 
eps = 1
alpha = 2
beta = 0.3

def random_V(K,N):
    V = np.random.normal(0,1,(K,N))
    V = V - V.mean(axis=1).reshape(-1,1)
    V = V / np.sqrt(np.sum(V**2,axis=1).reshape(-1,1))
    return V

def random_U(P,K,type='onegamma',alpha=2,beta=0.3):
    U = np.random.gamma(alpha,beta,(P,K))

def random_Y(U,V,eps):
    Y = U @ V + np.random.normal(0,eps/np.sqrt(N),(P,N)) 

    
def fit_data(U)
    


num=10
Uhat = np.empty((num,P,K))
Vhat = np.empty((num,K,N))
for i in range(num):
    # Determine random starting value 
    V_init = random_V(K,N)
    U_init = sparse_encode(Y,V_init,alpha=1,algorithm='lasso_cd')
    Uhat[i,:,:],Vhat[i,:,:],errors = dict_learning (Y,alpha = 1, n_components=5, method='cd',random_state=i,positive_code=True,code_init=U_init, dict_init=V_init)


