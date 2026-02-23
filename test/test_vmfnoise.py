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


def test_vmfnoise():
    """ Test vmf noise model"""
    K = 6
    P = 1000
    N = 4
    n_sub = 20
    n_part = 5

    # Make a random arrangement model
    ar_model = ar.ArrangeIndependent(K=K,P=P)
    ar_model.random_params()

    # Make a random emission model
    part_vec = pt.kron(pt.arange(n_part),pt.ones((N,)))
    cond_vec = pt.kron(pt.ones((n_part,)),pt.arange(N))
    X = ut.indicator(cond_vec)
    em_model = em.MixVMFNoise(K,N,P,n_sub,X=X,part_vec=part_vec)

    # Make data
    U = ar_model.sample(num_subj=n_sub)
    Y = em_model.sample(U)

    fm_fit = fm.FullMultiModel(ar_model,[em_model])

    fm_fit.initialize([Y])
    fm_fit,ll,th,Uhat= fm_fit.fit_em(iter=100, tol=0.01, fit_arrangement=True, fit_emission=True,first_evidence=False)

    pass



if __name__ == "__main__":
    test_vmfnoise()