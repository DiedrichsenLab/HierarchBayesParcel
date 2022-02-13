from arrangements import PottsModelGrid
from arrangements import PottsModelCortex
from emissions import MixGaussianExponential
import os # to handle path information
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
from nilearn import plotting
import surfAnalysisPy as surf

if __name__ == '__main__':
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
