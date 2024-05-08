#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2/18/2022
Full Model class

Author: dzhi, jdiedrichsen
"""
import numpy as np
import torch as pt
from torch.utils.data import Dataset, DataLoader
import HierarchBayesParcel.emissions as emi
import HierarchBayesParcel.arrangements as arr
import HierarchBayesParcel.evaluation as ev
import warnings
from copy import copy,deepcopy
import time
import pandas as pd

class FullMultiModel:
    """The full generative model contains arrangement model and multiple
       emission models for training across dataset
    """
    def __init__(self, arrange, emission):
        """Constructor

        Args:
            arrange (ArrangementModel):
                the arrangement model
            emission (list):
                the list of emission models
        """
        self.arrange = arrange
        self.emissions = emission
        self.n_emission = len(self.emissions)
        self.nparams = self.arrange.nparams + sum([i.nparams for i in self.emissions])
        self.K = self.emissions[0].K
        self.P = self.emissions[0].P

    def initialize(self,Y=None,subj_ind='separate'):
        """ Initializes the model for fitting.
        If Y or subj_ind is given, it replaces the existing.
        If set to None, the old existing will be used.

        Args:
            Y (list):
                List of (numsubj x N x numvoxel) arrays of data
            subj_ind (list):
                List of unique subject indicators OR
                'separate': sets seperate subjs for each data set OR
                None: Don't change anything
        """
        self.n_emission = len(self.emissions)
        self.nparams = self.arrange.nparams + sum([i.nparams for i in self.emissions])
        self.K = self.emissions[0].K
        self.P = self.emissions[0].P

        self.nsubj_list = []

        if subj_ind is not None:
            self.subj_ind = []

        # Got through emission models and initialize
        sbj = 0
        for i,em in enumerate(self.emissions):
            if Y is not None:
                em.initialize(Y[i])
            self.nsubj_list.append(em.num_subj)
            if isinstance(subj_ind,(list,np.ndarray)):
                if len(subj_ind[i])!=self.nsubj_list[i]:
                    raise NameError(f"length of subj_ind[{i}] does not match number of subjects:"
                                    f" {self.n_subj_list[i]}")
                else:
                    self.subj_ind.append(subj_ind[i])
            elif subj_ind == 'separate':
                self.subj_ind.append(pt.arange(sbj,self.nsubj_list[i]+sbj))
                sbj = sbj + self.nsubj_list[i]
            elif subj_ind is not None:
                raise(NameError("subj_ind needs to be an array/list, 'separate', or None"))
        # Overall number of unique subjects
        self.nsubj = max([max(i) for i in self.subj_ind]).item()+1
        self.ds_weight = pt.ones((self.n_emission,)) # Experimental dataset weighting

    def clear(self):
        """Clears the data from all emission models
        and temporary statistics from the arrangement model
        """
        self.arrange.clear()
        for em in self.emissions:
            em.clear()
        if hasattr(self,'self_ind'):
            delattr(self,'self_ind')

    def random_params(self, init_arrangement=True,
                      init_emission=True):
        """Sets all arrangement and emission model parameters to random values

        Args:
            init_arrangement (bool): Defaults to True.
            init_emission (bool): Defaults to True.
        """
        if init_arrangement:
            self.arrange.random_params()
        if init_emission:
            for em in self.emissions:
                em.random_params()

    def collect_evidence(self,emloglik):
        """Collects evidence over the different data sets
        For subjects that are in multiple datasets, it sums the
        log evidence.

        Args:
            emloglik (list): List of emissionlogliklihoods
        Returns:
            emloglik_comb (ndarray): ndarray of emission logliklihoods
        """
        if not hasattr(self,'subj_ind'):
            raise(NameError('subj_ind not found. Call model.initialize() first.'))
        emlog = pt.zeros(self.nsubj,self.K,self.P)
        for i,eml in enumerate(emloglik):
            emlog[self.subj_ind[i]]+=eml*self.ds_weight[i]
        return emlog

    def distribute_evidence(self,Uhat):
        """Splits the evidence to the different emission models

        Args:
            Uhat (pt.tensor): tensor of estimated or arrangement
        Returns:
            Usplit (list): List of Uhats (per emission model)
        """
        if not hasattr(self,'subj_ind'):
            raise(NameError('subj_ind not found. Call model.initialize() first.'))
        Usplit = []
        for i,s in enumerate(self.subj_ind):
            Usplit.append(Uhat[s])
        return Usplit

    def marginal_prob(self):
        """Convenience function that returns
        marginal probability for the arrangement model

        Returns:
            Prob (pt.tensor): KxP marginal probabilities
        """
        return self.arrange.marginal_prob()

    def set_num_subj(self,num_subj=None):
        """Sets the number of subjects for simulations

        Args:
            num_subj (list):
                list of subjects number. i.e [2, 3, 4] for each dataset OR
                list of subject indices [[0,1,2],[0,1,2],[2,3,4]] for overlapping subjects
        """
        # Default - assign 10 subjects per dataset
        if num_subj is None:
            num_subj = [10] * len(self.emissions)

        # Independent subjects in all data sets
        if isinstance(num_subj[0],int):
            for i,em in enumerate(self.emissions):
                em.num_subj = num_subj[i]
            self.initialize(Y=None,subj_ind='separate')
        # Specific overlap between data sets
        elif isinstance(num_subj[0],(list,np.ndarray,pt.tensor)):
            for i,em in enumerate(self.emissions):
                em.num_subj = len(num_subj[i])
            self.initialize(Y=None,subj_ind=num_subj)
        else:
            raise NameError('num_subj needs to be a list of ints or a list of array/lists')

    def sample(self, num_subj=None, U=None):
        """Take in the number of subjects to sample for each emission model

        Args:
            num_subj (list):
                list of subjects number. i.e [2, 3, 4] Or
                list of subject indices [[0,1,2],[0,1,2],[2,3,4]]
        Returns:
            U: the true Us of all subjects concatenated vertically,
               shape(num_subs, P)
            Y: data sampled from emission models, shape (num_subs, N, P)
        """
        # If number of subject is given or current nsubj in the model is None,
        # then we overwrite the
        # subjects data per each emission model
        if (num_subj is not None) or (self.nsubj is None):
            self.set_num_subj(num_subj)

        if U is None:
            U = self.arrange.sample(self.nsubj)
        Y = []

        for em, Us in enumerate(self.distribute_evidence(U)):
            this_Y = self.emissions[em].sample(Us, signal=None)
            Y.append(this_Y)
        return U, Y

    def Estep(self, separate_ll=False):
        """E step for full model. Run a full process of EM procedure once on both arrangement and emission models.

        Args:
            separate_ll: if True, return separte ll_A and ll_E
        Returns:
            Uhat: the prediction U_hat
            ll: the log-likelihood for arrangement and emission models
        """
        # Run E-step
        emloglik = [e.Estep() for e in self.emissions]

        # Collect the evidence and broadcast to arrangement mode
        emloglik_comb = self.collect_evidence(emloglik)  # combine the log-liklihoods
        Uhat, ll_a = self.arrange.Estep(emloglik_comb)
        ll_e = (Uhat * emloglik_comb).sum()

        ar_name = type(self.arrange).__name__
        if separate_ll:
            return Uhat, [ll_e, ll_a.sum()] \
                if ar_name.startswith('ArrangeIndependent') else [ll_e, pt.nan]
        else:
            return Uhat, ll_a.sum()+ll_e \
            if ar_name.startswith('ArrangeIndependent') else pt.nan

    def fit_em(self,iter=30, tol=0.01, seperate_ll=False, fit_emission=True,
               fit_arrangement=True, first_evidence=False):
        """ Run the EM-algorithm on a full model this demands that both the Emission and Arrangement model
            have a full Estep and Mstep and can calculate the likelihood, including the partition function

        Args:
            iter (int): Maximal number of iterations (def:30)
            tol (double): Tolerance on overall likelihood (def: 0.01)
            seperate_ll (bool): Return arrangement and emission LL separetely
            fit_emission (list / array of bools): If True, fit emission model.
                    Otherwise, freeze it
            fit_arrangement: If True, fit the arrangement model.
                    Otherwise, freeze it
            first_evidence (bool or list of bool): Determines whether evidence
                    is passed from emission models to arrangement model on the
                    first iteration. Default = False. This is improve estimation for
                    emission models from random starting values, as the initial guess
                    will be determined by the intialization of the arrangement model.
                    If a list of bools, it determines this for each emission model seperately.
        Returns:
            model (Full Model): fitted model (also updated)
            ll (ndarray): Log-likelihood of full model as function of iteration
                If seperate_ll, the first column is ll_A, the second ll_E
            theta (ndarray): History of the parameter vector
            Uhat (pt.tensor): (n_subj,K,P) matrix of estimates - note that
                this is in the space of arrangement model - call
                distribute_evidence(Uhat) to get this in the space of
                emission model
        """
        if not hasattr(fit_emission, "__len__"):
            fit_emission = [fit_emission]*len(self.emissions)
        if not hasattr(first_evidence, "__len__"):
            first_evidence = [first_evidence]*len(self.emissions)

        # Initialize the tracking
        ll = pt.zeros((iter, 2))
        theta = pt.zeros((iter, self.nparams))

        # Run number of iterations
        for i in range(iter):
            # Track the parameters
            theta[i, :] = self.get_params()

            # Get the (approximate) posterior p(U|Y)
            # emloglik = [e.Estep() for e in self.emissions]
            # Pass emlogliks immediately to collect evidence function,
            # rather than save a local variable `emloglik` to waste memory
            # If first iteration, only pass the desired emission models (pretraining)
            emloglik_c = [e.Estep() for e in self.emissions]
            pt.cuda.empty_cache()
            if i==0:
                for j,emLL in enumerate(emloglik_c):
                    if not first_evidence[j]:
                        emLL[:,:,:]=0
            emloglik_comb = self.collect_evidence(emloglik_c)  # Combine subjects
            del emloglik_c
            pt.cuda.empty_cache()

            Uhat, ll_A = self.arrange.Estep(emloglik_comb)
            # Compute the expected complete logliklihood
            ll_E = pt.sum(Uhat * emloglik_comb, dim=(1, 2))
            del emloglik_comb
            pt.cuda.empty_cache()

            ll[i, 0] = pt.sum(ll_A)
            ll[i, 1] = pt.sum(ll_E)
            if pt.isnan(ll[i,:].sum()):
                raise(NameError('Likelihood returned a NaN'))
            # Check convergence:
            # This is what was here before. It ignores whether likelihood increased or decreased!
            # if i == iter - 1 or ((i > 1) and (np.abs(ll[i, :].sum() - ll[i - 1, :].sum()) < tol)):
            if i==iter-1:
                break
            elif i>1:
                dl = ll[i,:].sum()-ll[i-1,:].sum() # Change in logliklihood
                # Check if likelihood decreases more than tolerance
                if dl<-tol:
                    warnings.warn(f'Likelihood decreased - terminating on iteration {i}')
                    break
                elif dl<tol:
                    break

            # Updates the parameters
            Uhat_split = self.distribute_evidence(Uhat)
            if fit_arrangement:
                self.arrange.Mstep()
            for em, Us in enumerate(Uhat_split):
                if fit_emission[em]:
                    self.emissions[em].Mstep(Us)

            pt.cuda.empty_cache()

        # Clear the temporary stats from the arrangement model to concerve memory
        self.arrange.clear()
        pt.cuda.empty_cache()

        # Return parameters and Uhat
        if seperate_ll:
            return self, ll[0:i+1], theta[0:i+1, :], Uhat
        else:
            return self, ll[0:i+1].sum(axis=1), theta[0:i+1, :], Uhat

    def fit_em_ninits(self, n_inits=20, first_iter=7, iter=30, tol=0.01,
                      fit_emission=True, fit_arrangement=True,
                      init_emission=True, init_arrangement=True,
                      align = 'arrange',verbose=True):
        """Run the EM-algorithm on a full model starting with n_inits multiple random initialization values and escape from local maxima by selecting the model with the highest likelihood after first_iter. This demands that both the Emission and Arrangement model have a full Estep and Mstep and can calculate the likelihood, including the partition function

        Args:
            n_inits: the number of random inits
            first_iter: the first few iterations for the random inits to find
                        the inits parameters with maximal likelihood
            iter: the number of iterations for full EM process
            tol: Tolerance on overall likelihood (def: 0.01)
            fit_emission (list): If True, fit emission model. Otherwise, freeze it
            fit_arrangement: If True, fit arrangement model. Otherwise, freeze it
            init_emission (list or bool): Randomly initialize emission models before fitting?
            init_arrangement (bool): Randomly initialize arrangement model before fitting?
            align: (None,'arrange', or int): Alignment one first step is performed
                None: Not performed - Emission models may not get aligned
                'arrange': from the arrangement model only (works only if spatially non-flat (i.e. random) initialization)
                int: from emission model with number i. Works with spatially flat initialization of arrangement model
            verbose: if set to true, gives memory update for each iteration
        Returns:
            model (Full Model): fitted model (also updated)
            ll (ndarray): Log-likelihood of best full model as function of iteration
                the initial iterations are included
            theta (ndarray): History of the parameter vector
            Uhat: the predicted U (probabilistic)
            first_lls: the log-likelihoods for the n_inits random parameters
                       after first_iter runs
        """
        max_ll = pt.tensor([-pt.inf])
        first_lls = pt.full((n_inits,first_iter), pt.nan)
        # Set the first passing of evidence based on alignment pretraining:
        if align is None:
            first_ev = [True]*len(self.emissions)
        elif align=='arrange':
            first_ev = [False]*len(self.emissions)
        else:
            first_ev = [False]*len(self.emissions)
            first_ev[align]=True
        if verbose:
            print('n_inits starting')
            report_cuda_memory()

        for i in range(n_inits):
            # Making the new set of emission models with random initializations
            fm = deepcopy(self)
            if verbose:
                print(f'{i} n inits')
                report_cuda_memory()
            if init_arrangement:
                fm.arrange.random_params()
            if init_emission:
                for em in fm.emissions:
                    em.random_params()

            fm, this_ll, theta, _ = fm.fit_em(first_iter, tol=tol,
                                              seperate_ll=False,
                                              fit_emission=fit_emission,
                                              fit_arrangement=fit_arrangement,
                                              first_evidence=first_ev)
            first_lls[i,:len(this_ll)]=this_ll
            if this_ll[-1] > max_ll[-1]:
                max_ll = this_ll
                self = fm
                best_theta = theta

        self, ll, theta, U_hat = self.fit_em(iter=iter-first_iter, tol=tol,
                                             seperate_ll=False,
                                             fit_emission=fit_emission,
                                             fit_arrangement=fit_arrangement,
                                             first_evidence=True)
        ll_n = pt.cat([max_ll,ll[1:]])
        theta_n = pt.cat([best_theta,theta[1:, :]])
        return self, ll_n, theta_n, U_hat, first_lls

    def fit_sml(self, iter=60, batch_size=None, stepsize=0.8, estep='sample',
                seperate_ll=False, fit_emission=True, fit_arrangement=True,
                first_evidence=True):
        """ Runs a Stochastic Maximum likelihood algorithm on a full model.
        The emission model is still assumed to have E-step and Mstep.
        The arrangement model is has a postive and negative phase estep,
        and a gradient M-step. The arrangement likelihood is not necessarily
        FUTURE EXTENSIONS:
        * Sampling of subjects from training set
        * initialization of parameters
        * adaptitive stopping criteria
        * Adaptive stepsizes
        * Gradient acceleration methods

        Args:
            Y (3d-ndarray): numsubj x N x numvoxel array of data
            iter (int): Maximal number of iterations
            stepsize (double): Fixed step size for MStep
        Returns:
            model (Full Model): fitted model (also updated)
            ll (ndarray): Log-likelihood of full model as function of iteration
                If seperate_ll, the first column is ll_A, the second ll_E
            theta (ndarray): History of the parameter vector
        """
        if not hasattr(fit_emission, "__len__"):
            fit_emission = [fit_emission]*len(self.emissions)
        if not hasattr(first_evidence, "__len__"):
            first_evidence = [first_evidence]*len(self.emissions)

        # Initialize the tracking
        self.arrange.alpha = stepsize
        ll = pt.zeros((iter,2))
        theta = pt.zeros((iter,))

        for i in range(iter):
            print(f'start: {i}')
            # Track the parameters
            if 'theta' in self.arrange.param_list:
                theta[i] = self.arrange.theta

            # Get the (approximate) posterior p(U|Y)
            emloglik_c = [e.Estep() for e in self.emissions]
            pt.cuda.empty_cache()
            if i==0:
                # Align emission models at the first iteration by
                # choosing which emloglik to be passed-up
                for j in range(len(emloglik_c)):
                    if not first_evidence[j]:
                        emloglik_c[j].zero_()
                # for j,emLL in enumerate(emloglik_c):
                #     if not first_evidence[j]:
                #         emLL[:,:,:]=0
            emloglik_comb = self.collect_evidence(emloglik_c)  # Combine subjects
            del emloglik_c
            pt.cuda.empty_cache()

            if batch_size is None:
                batch_size = find_maximum_divisor(emloglik_comb.shape[0])

            # Create a random permutation of indices for sampling
            sampled_indices = pt.randperm(len(emloglik_comb))
            sampler = pt.utils.data.SubsetRandomSampler(sampled_indices)

            # Create a DataLoader object for training data
            train_loader = DataLoader(fMRI_Dataset(emloglik_comb),
                                      batch_size=batch_size,
                                      sampler=sampler,
                                      num_workers=0)
            del emloglik_comb
            pt.cuda.empty_cache()

            # Update the arrangment model in batches
            for j, (bat_emlog_train, bat_indx) in enumerate(train_loader):
                print(f'------Batch {j+1}: training batch size '
                      f'{bat_emlog_train.shape}, batch subject '
                      f'indices {bat_indx} ------')
                # 1. arrangement E-step: positive phase
                tic = time.perf_counter()
                self.arrange.Estep(bat_emlog_train)
                toc = time.perf_counter()
                print(f'positive phase {self.arrange.epos_iter} '
                      f'iters used {toc - tic:0.4f} seconds!')
                pt.cuda.empty_cache()
                report_cuda_memory()

                # 2. arrangement E-step: negative phase
                tic = time.perf_counter()
                if hasattr(self.arrange, 'Eneg'):
                    # TODO: if there are multiple emission models,
                    # which emission should be used for sampling?
                    self.arrange.Eneg(use_chains=bat_indx,
                                      emission_model=deepcopy(self.emissions[0]))
                toc = time.perf_counter()
                print(f'negative phase {self.arrange.eneg_iter} '
                      f'iters used {toc - tic:0.4f} seconds!')
                pt.cuda.empty_cache()
                report_cuda_memory()

                if fit_arrangement:
                    # 3. arrangement M-step
                    tic = time.perf_counter()
                    self.arrange.Mstep()
                    toc = time.perf_counter()
                    print(f'M-step used {toc - tic:0.4f} seconds!')
                    pt.cuda.empty_cache()
                    report_cuda_memory()

            # Monitor the RBM training - cross entropy
            CE = ev.cross_entropy(self.arrange.epos_Uhat,
                                  self.arrange.eneg_U)
            # Compute Uhat in batch - Don't gather the sufficient
            # statistics as the model is already updated
            Uhat = []
            for b in range(0, self.nsubj - batch_size + 1, batch_size):
                ind = range(b, b + batch_size)
                Uhat.append(self.arrange.Estep(train_loader.dataset.data[ind,:,:],
                                             gather_ss=False)[0])
            Uhat = pt.vstack(Uhat)
            pt.cuda.empty_cache()

            # Compute the expected emission logliklihood
            ll_E = pt.sum(Uhat * train_loader.dataset.data, dim=(1, 2))
            ll[i, 0] = -pt.sum(CE) # negative entropy as a penalty term
            ll[i, 1] = pt.sum(ll_E)
            if pt.isnan(ll[i,:].sum()):
                raise(NameError('Likelihood returned a NaN'))

            # TODO: contruct a convergence criterion?
            if i==iter-1:
                break

            # Updates the parameters
            Uhat_split = self.distribute_evidence(Uhat)
            for em, Us in enumerate(Uhat_split):
                if fit_emission[em]:
                    # self.emissions[em].initialize(Us)
                    self.emissions[em].Mstep(Us)

            # clear cache by the end of each iteration
            pt.cuda.empty_cache()

        # Clear the temporary stats from the arrangement model to concerve memory
        self.arrange.clear()
        pt.cuda.empty_cache()

        if seperate_ll:
            return self, ll[0:i+1], theta, Uhat
        else:
            return self, ll[0:i+1].sum(axis=1), theta, Uhat


    def get_params(self):
        """Get the concatenated parameters from arrangemenet + emission model

        Returns:
            theta (ndarrap)
        """
        emi_params = [i.get_params() for i in self.emissions]
        return pt.cat([self.arrange.get_params(), pt.cat(emi_params)])

    def get_param_indices(self, name):
        """Return the indices for the full model theta vector

        Args:
            name (str): Parameter name in the format of 'arrange.logpi'
                        or 'emissions.<X>.V' where <X> is the index of
                        emission model. For example 'emissions.0.V' will
                        return the Vs from self.emissions[0]
        Returns:
            indices (np.ndarray): 1-d numpy array of indices into the theta vector
        """
        names = name.split(".")
        if (len(names) == 2) and (names[0] == 'arrange'):
            ind = vars(self)[names[0]].get_param_indices(names[1])
            return ind
        elif (len(names) == 3) and (names[0] == 'emissions'):
            ems_nparams = np.cumsum([i.nparams for i in self.emissions])
            nparams_offset = np.insert(ems_nparams, 0, 0) + self.arrange.nparams
            ind = vars(self)[names[0]][int(names[1])].get_param_indices(names[2])
            return ind + nparams_offset[int(names[1])]
        else:
            raise NameError('Parameter name needs to be arrange.<param> '
                            'or emissions.<X>.<param>, where <X> is the index '
                            'of emission model and <param> is the param name. i.g '
                            'emissions.0.V')

    def move_to(self, device='cpu'):
        """Recursively move all torch.Tensor object in fullModel
           class to the targe device

        Args:
            M (FullMultiModel): Full model
            device (str or pt.device):
                the target device to store the tensors default - 'cpu'
       """
        for attr, value in self.__dict__.items():
            if isinstance(value, pt.Tensor):
                vars(self)[attr] = value.to(device)
            elif attr == 'arrange':
                value.move_to(device=device)
            elif attr == 'emissions':
                for obj_in_list in value:
                    obj_in_list.move_to(device=device)


class fMRI_Dataset(Dataset):
    """Helper class to create mini-batches from a large tensor
    """
    def __init__(self, data):
        """Constructor for the dataset
        Args:
            data: tensor of data
        """
        self.data = data

    def __len__(self):
        """Returns the length of the dataset
        """
        return len(self.data)

    def __getitem__(self, index):
        """Returns the data at the given index
        Args:
            index: index of the data

        Returns:
            data: data at the given index
            index: index of the data
        """
        data = self.data[index]
        return data, index

def prep_datasets(dat, info, cond_vector, part_vector, join_sess=False,
                  join_sess_part=False):
    """ Builds dataset, cond_vec, part_vec, subj_ind from the given dataset
        in Functional fusion project.

    Args:
        dat (numpy.ndarray or pytorch.Tensor):
            Input data tensor, must be in shape of n_subj x n_cond x n_voxels.
        info (numpy.ndarray or pandas.DataFrame):
            An information pandas.Series or numpy.ndarray to indicate how to
            split the data into different parts based on the unique values in
            the info vector.
        cond_ind (1d numpy.ndarray):
            The condition vector used to make the design matrix for indicating
            the unique conditions associated to an emission model.
        part_ind (1d numpy.ndarray):
            The partition vector tells a task measure is from which session.
        join_sess (boolean):
            The multiple sessions will be modeled using a single emission model.
            Defaults to True which corresponds to model 01, 02, 05. If set to
            False, the emission model will model the sessions separately, such
            as the model 03 and 04.
        join_sess_part (boolean):
            If set to True, the partition vector will be set to 1 for all the
            data points, which means no partitions in the data and all repeats
            of task measures are concatenated as a long vector. Defaults to False.

    Returns:
        data (list): A list of the data tensor
        cond_vec (list): A list of the condition vector
        part_vec (list): A list of the partition vector
        subj_ind (list): A list of the subject index

    Notes:
        The returned data, cond_vec, part_vec, subj_ind will have the same
        length. The elements in the lists are indexly matched to model an
        emission model.
    """
    # Check and convert info pandas Series to NumPy arrays
    if isinstance(info, pd.Series):
        info = info.values

    sub = 0
    data, cond_vec, part_vec, subj_ind = [], [], [], []
    n_subj = dat.shape[0]

    # Make different sessions either the same or different
    if join_sess:
        data.append(dat)
        cond_vec.append(cond_vector.reshape(-1, ))

        # Check if we want to set no partition after join sessions
        if join_sess_part: # Model 05
            part_vec.append(np.ones(part_vector.shape))
        else:
            part_vec.append(part_vector.reshape(-1, ))
        subj_ind.append(np.arange(sub, sub + n_subj))
    else:
        sessions = np.unique(info)
        # Now build and split across the correct sessions:
        for s in sessions:
            indx = info == s
            data.append(dat[:, indx, :])
            cond_vec.append(cond_vector[indx].reshape(-1, ))
            part_vec.append(part_vector[indx].reshape(-1, ))
            subj_ind.append(np.arange(sub, sub + n_subj))

    return data, cond_vec, part_vec, subj_ind

def get_indiv_parcellation(ar_model, atlas, train_data, cond_vec, part_vec,
                           subj_ind, Vs=None, sym_type='asym', n_iter=200,
                           em_params={},
                           fit_arrangement=False,
                           fit_emission=True, device=None):
    """ Calculates the individual parcellations using the given individual
        training data and the given arrangement model with the pre-defined
        group prior.

    Args:
        ar_model (arrangement model object):
            The arrangement model object with pre-defined group prior U.
        atlas (object):
            The atlas object for the arrangement model
        train_data (np.ndarray or pt.Tensor):
            Individual localizing data
        cond_vec (list):
            The condition vectors for each emission model
        part_vec (list):
            The partition vectors for each emission model
        subj_ind (list):
            The subject indices for each emission model
        Vs (list):
            The mean response vectors for each emission model, if None, the
            mean response vectors will be calculated from random inits.
            If not None, the Vs should be a list of the V vectors for each
            emission model, and V will be fixed during the learning.
        sym_type (str):
            The symmetry type of the arrangement model
        n_iter (int):
            The number of iterations for the EM algorithm
        em_params (dictionary):
            Dictionary setting optina parameters for the emission model
        fit_arrangement (boolean):
            If True, the arrangement model will be fitted using the given
            individual training data. However, in this case, the arrangement
            model should be freezed during the learning process.
        fit_emission (boolean):
            If True, the emission models will be fitted. The emission model
            parameters are freely learned.
        device (str):
            The device name to load trained model

    Returns:
        U_indiv (pt.Tensor):
            The individual probabilistic parcellations
        ll (list):
            The log-likelihood of the individual parcellations
        M (object):
            The trained arrangement model
    """
    # convert tdata to tensor
    if type(train_data) is np.ndarray:
        train_data = pt.tensor(train_data, dtype=pt.get_default_dtype())
    if Vs is None:
        Vs = [None] * len(train_data)

    # Check if the lists have equal length using assert
    assert len(train_data) == len(cond_vec) == len(part_vec) == len(Vs),\
        "training data, condition vector, and partition vector " \
        "must have equal length."

    # Check if the input arrangement model is valid
    if not isinstance(ar_model, arr.ArrangementModel):
        raise ValueError("The input model must be a valid arrangement"
                         " model object")

    # Initialize emission models
    em_models = []
    for j, this_cv in enumerate(cond_vec):
        if sym_type=='sym':
            K=ar_model.K_full
        else:
            K=ar_model.K
        em_model = emi.build_emission_model(K,
                                            atlas,
                                            'VMF',
                                            indicator(this_cv),
                                            part_vec[j],
                                            V=Vs[j],
                                            em_params=em_params)
        em_models.append(em_model)

    M = FullMultiModel(ar_model, em_models)
    M.initialize(train_data, subj_ind=subj_ind)

    # ---------------------------------------------------------
    # Real training starts here with a frozen arrangement model
    # ---------------------------------------------------------
    if M.arrange.name.startswith('indp'):
        M, ll, _, U_indiv = M.fit_em(iter=n_iter, tol=0.01,
                                     fit_arrangement=fit_arrangement,
                                     fit_emission=fit_emission,
                                     first_evidence=False)
    else:
        raise NameError("The arrangement model is not supported yet.")

    # Return the individual PROBABILISTIC parcellations
    return U_indiv, ll, M
