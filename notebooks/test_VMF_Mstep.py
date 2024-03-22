#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests different M-step methods for the VMF distribution
In this script, we follow a very simple example where there is only one possible parcel 
and we can estimate V and kappa directly in one M-step.    
"""

# global package import
from copy import copy, deepcopy
import pandas as pd
import seaborn as sb
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

# for testing and evaluating models
from HierarchBayesParcel.full_model import FullModel, FullMultiModel
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.spatial as sp
import HierarchBayesParcel.evaluation as ev
from sklearn.metrics.pairwise import cosine_similarity


def make_data(): 


def test_subject_differences(): 
    