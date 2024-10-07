#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The script to test load fMRI dataset using pyTorch default
DataLoader class.

Created on 3/15/2023 at 10:06 PM
Author: dzhi
"""
import os
import pandas as pd
from pathlib import Path
import torch as pt
from torch.utils.data import Dataset, DataLoader
import numpy as np

pt.set_default_tensor_type(pt.cuda.FloatTensor
                           if pt.cuda.is_available() else
                           pt.FloatTensor)

# Find model directory to save model fitting results
MODEL_DIR = 'Y:/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(MODEL_DIR).exists():
    MODEL_DIR = '/srv/diedrichsen/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(MODEL_DIR).exists():
    MODEL_DIR = '/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(MODEL_DIR).exists():
    raise (NameError('Could not find MODEL_DIR'))

BASE_DIR = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(BASE_DIR).exists():
    BASE_DIR = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(BASE_DIR).exists():
    BASE_DIR = 'Y:/data/FunctionalFusion'
if not Path(BASE_DIR).exists():
    BASE_DIR = '/Users/callithrix/Documents/Projects/Functional_Fusion/'
if not Path(BASE_DIR).exists():
    BASE_DIR = '/Users/jdiedrichsen/Data/FunctionalFusion/'
if not Path(BASE_DIR).exists():
    raise (NameError('Could not find BASE_DIR'))

class fMRI_Dataset(Dataset):
    def __init__(self, name, smooth=None):
        self.data_dir = BASE_DIR + f'/{name}/derivatives'
        self.part_info = pd.read_csv(BASE_DIR + f'/{name}/participants.tsv', delimiter='\t')
        subject_list = self.part_info.participant_id.to_numpy()
        self.samples = []
        for subject_dir in subject_list:
            if not os.path.isdir(os.path.join(self.data_dir, subject_dir)):
                raise ValueError(f'No {subject_dir} found in the derivative folder!')

            if smooth is not None:
                run_dir = os.path.join(self.data_dir, subject_dir, f'data/smoothed_{smooth}mm')
            else:
                run_dir = os.path.join(self.data_dir, subject_dir, 'data')

            # load data from file or directory and preprocess as necessary
            data = np.load(os.path.join(self.data_dir, subject_dir, run_dir, 'data.npy'))
            labels = np.load(os.path.join(self.data_dir, subject_dir, run_dir, 'labels.npy'))
            # add the data and labels to the list of samples
            self.samples.append((data, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data, labels = self.samples[index]
        # perform any additional preprocessing on the data and labels
        data = pt.tensor(data)
        labels = pt.tensor(labels)
        return data, labels


if __name__ == '__main__':
    dataset = fMRI_Dataset(name='MDTB')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)