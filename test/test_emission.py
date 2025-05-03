import HierarchBayesParcel.evaluation as ev
import HierarchBayesParcel.arrangements as ar
import Functional_Fusion.dataset as ds
import torch as pt
import nibabel as nb
import nitools as nt
import pandas as pd
import matplotlib.pyplot as plt
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.util as ut
import numpy as np

# base Dir for function fusion data
base_dir  = '/Volumes/diedrichsen_data$/data/FunctionalFusion'


def test_emission():
    """ Test estimation of kappa in emission model"""
    K = 4
    P = 1000
    N = 4
    n_sub = 20
    n_part = 5

    # Make a random arrangement model
    ar_true = ar.ArrangeIndependent(K=K,P=P)
    ar_true.random_params()

    # Make a random emission model
    part_vec = pt.kron(pt.arange(n_part),pt.ones((N,)))
    cond_vec = pt.kron(pt.ones((n_part,)),pt.arange(N))
    X = ut.indicator(cond_vec)
    em_true = em.MixVMF(K,N,P,n_sub,X=X,part_vec=part_vec)
    em_true.random_params()
    em_true.kappa = pt.tensor([7.0])

    # Make data
    U = ar_true.sample(num_subj=n_sub)
    Y = em_true.sample(U)

    # Make emission model for fitting
    em_fit = em.MixVMF(K,N,P,n_sub,X=X,part_vec=part_vec)
    em_fit.random_params()
    em_fit.kappa = pt.tensor([3.0])
    fm_fit = fm.FullMultiModel(ar_true,[em_fit])

    fm_fit.initialize([Y])
    fm_fit,ll,th,Uhat= fm_fit.fit_em(iter=100, tol=0.01, fit_arrangement=False, fit_emission=True,first_evidence=False)

    # plot the results
    ki = fm_fit.get_param_indices('emissions.0.kappa')
    Vi = fm_fit.get_param_indices('emissions.0.V')
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(th[:,ki])
    plt.subplot(2,2,3)
    plt.plot(th[:,Vi])

    # Check that the V is estimated correctly
    plt.subplot(2,2,2)
    plt.imshow(em_true.V.t() @ em_true.V)
    plt.colorbar()
    pass



if __name__ == "__main__":
    test_emission()