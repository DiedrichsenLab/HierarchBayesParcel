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


def test_cosine_error():
    """ Test different forms of the cosine error function """
    N = 7
    K = 4
    P = 1000
    n_sub = 3
    # Make a V matrix
    V = pt.randn((N, K))
    V = V / V.norm(dim=0,keepdim=True)

    # Make a U matrix
    U = pt.rand((n_sub,K,P))
    U = U / U.sum(dim=1,keepdim=True)
    Uhard = ar.expand_mn(ar.compress_mn(U),K)

    # Make different datasets
    Y_rand = pt.randn((n_sub,N, P))
    Y_hard = pt.matmul(V,Uhard)
    Y_avrg = pt.matmul(V,U)
    type = ['hard','average','expected']
    ytyp = ['random','hard','average']


    # Test 1: different simulation scenarios
    for i,Y in enumerate([Y_rand, Y_hard, Y_avrg]):
        # Test the cosine error
        for t in type:
            error = ev.coserr(Y,V,U, type=t,adjusted=False)
            print(f"{t} cos error on {ytyp[i]} data: {error}")
    pass

    # Test 2: Different sizes
    error = ev.coserr(Y_avrg,V,U[0,:,:], type='average',adjusted=False)
    print(f"cos error with 1 prediction: {error}")
    error = ev.coserr(Y_avrg[0,:,:],V,U[0,:,:], type='average',adjusted=False)
    print(f"cos error with 1 size: {error}")
    pass

def test_calc_test_error():
    # Get atlas
    subjs = np.arange(10)
    atlas, _ = am.get_atlas('MNISymC3')
    # Sample the probabilistic atlas at the specific atlas grayordinates
    atlas_fname = f'/Users/callithrix/code/Python/HierarchBayesParcel/examples/atl-NettekovenSym32_space-MNI152NLin2009cSymC_probseg.nii.gz'
    U = atlas.read_data(atlas_fname)
    U = U.T
    # Build the arrangement model - the parameters are the log-probabilities of the atlas
    ar_model = ar.build_arrangement_model(U, prior_type='prob', atlas=atlas)

    # Get the training data
    data_train, info_train, _ = ds.get_dataset(dataset='MDTB', base_dir=base_dir,atlas='MNISymC3',
                                               sess='ses-s1',type='CondHalf',subj=subjs)

    # K is the number of parcels
    K = ar_model.K
    # Make a design matrix
    X= ut.indicator(info_train['cond_num'])
    # Build an emission model
    em_model_train = em.MixVMF(K=K,P=atlas.P, X=X,part_vec=info_train['half'])

    # Build the full model: The emission models are passed as a list, as usually we have multiple data sets
    M = fm.FullMultiModel(ar_model, [em_model_train])

    # Attach the data to the model - this is done for speed
    M.initialize([data_train])

    # Now we can run the EM algorithm to estimate emission model and get individual parcellations
    M, ll, _, U_hat = M.fit_em(iter=200, tol=0.01,
        fit_arrangement=False,fit_emission=True,first_evidence=False)

    # Get the training data
    data_test, info_test, _ = ds.get_dataset(dataset='MDTB', base_dir=base_dir,atlas='MNISymC3',
                                             sess='ses-s2',type='CondHalf',subj=subjs)
    # Build an emission model for test data
    X= ut.indicator(info_test['cond_num'])
    em_model_test = em.MixVMF(K=K,P=atlas.P, X=X,part_vec=info_test['half'])
    Mt = fm.FullMultiModel(ar_model, [em_model_test])

    cos_err1 = ev.calc_test_error(Mt, data_test, [U_hat,'group'], coserr_type='average',coserr_adjusted=True, fit_emission='full')
    cos_err2 = ev.calc_test_error(Mt, data_test, [U_hat,'group'], coserr_type='average',coserr_adjusted=True, fit_emission='use_Uhats')
    cos_err3 = ev.calc_test_error(Mt, data_test, [U_hat,'group'], coserr_type='average',coserr_adjusted=True, fit_emission='use_Uhats')
    print(cos_err1.mean(axis=1))
    print(cos_err2.mean(axis=1))
    sb.barplot(data=pd.DataFrame({'full':cos_err1.mean(axis=1),'Uhat':cos_err2.mean(axis=1)}))
    pass

if __name__ == "__main__":
    test_calc_test_error()