from arrangements import ArrangeIndependent
from emissions import MixGaussianExponential
import os # to handle path information
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
from nilearn import plotting
import surfAnalysisPy as surf

class fullModel: 
    def __init__(self,arrange,emission):
        self.arrange = arrange
        self.emission = emission
    
    def sample(self,num_subj=10):
        U = self.arrange.sample(num_subj)
        Y = self.emission.sample(U)
        return Y,U
    
    def fit_em(self,Y,iter,tol):
        # Initialize the tracking 
        ll = np.zeros((iter,))
        theta  = np.zeros((iter,self.emission.nparams+self.arrange.nparams))
        for i in range(iter): 
            # Get the (approximate) posterior p(U|Y)
            emloglik = self.emission.Estep(Y)
            Uhat,ll_A = self.arrange.Estep(emloglik)
            # Compute the expected complete logliklihood 
            ll[i] = np.sum(Uhat * emloglik) + ll_A
            # Updates the parameters 
            self.emission.Mstep(Uhat)
            self.arrange.Mstep(Uhat)
            theta[i,:]=np.concatenate([self.emission.get_params(),
                                         self.arrange.get_params()])
        return self,theta,ll


def _fit_full(Y):

def _simulate_full():
    arrangeT = ArrangeIndependent(K=5,P=100,spatial_specific=False)
    emissionT = MixGaussianExponential(K=5,N=40,P=100)
    # Set the true parameters to some interesting value

    # Generate data 
    U = arrangeT.sample(num_subj=10)
    Y = emissionT.sample(U)

    # Generate new models for fitting 
    arrangeM = ArrangeIndependent(K=5,P=100,spatial_specific=False)
    emmisionM = MixGaussianExponential(K=5,N=40,P=100)
    
    em_algorithm(arrangeM,emmisionM,Y)


if __name__=="__main__":
    _simulate_full() 

    M = PottsModelCortex(17,roi_name='Icosahedron-1442')
    M.define_mu(2000)
    cluster = np.argmax(M.mu,axis=0)+1
    cl = M.map_data(cluster)
    data = cl[0]
    # U = M.generate_subjects(num_subj = 4)
    # [Y,param] = M.generate_emission(U)
    # pass
    num_subj = 3 
    U = np.zeros((num_subj,M.P))
    for i in range(num_subj):
        Us = M.sample_gibbs(prior=True,iter=10)
        U[i,:]=Us[-1,:]
    for i in range(num_subj):
        s = M.map_data(U[i,:]+1)
        data = np.c_[data,s[0]]
    # surf.map.make_label_gifti()
    MAP = plt.get_cmap('tab20')
    RGBA = MAP(np.arange(18))
    G = surf.map.make_label_gifti(data,label_RGBA=RGBA)
    nb.save(G,'indivmaps.label.gii')
