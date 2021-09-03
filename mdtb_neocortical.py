#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements functions related to calculating and handeling cortical maps 

@author: jdiedrichsen
"""
import os # to handle path information
import nibabel as nb
import numpy as np
import h5py
import pandas as pd
import surfAnalysisPy as surf
import matplotlib.pyplot as plt

return_subjs = np.array([2,3,4,6,8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31])

baseDir = '/Volumes/diedrichsen_data$/data/super_cerebellum'
surfDir = baseDir + '/sc1/surfaceWB'

hem = ['L','R']
hem_name = ['CortexLeft','CortexRight']


def load_surf():
    flatsurf = []
    inflsurf = [] 

    for h,hemN in enumerate(hem):
        flatname = os.path.join(surfDir,'group32k','fs_LR.32k.' + hemN + '.flat.surf.gii')
        inflname = os.path.join(surfDir,'group32k','fs_LR.32k.' + hemN + '.inflated.surf.gii')
        flatsurf.append(nb.load(flatname))
        inflsurf.append(nb.load(flatname))
    return(flatsurf,inflsurf)

def load_wcon(subj='s02'): 
    subj = 's02'
    taskmap = []
    for h,hemN in enumerate(hem):
        name = os.path.join(surfDir,'glm7',subj,subj + '.' + hemN + '.sc1.wcon.func.gii')
        taskmap.append(nb.load(name))
    colnames = surf.map.get_gifti_column_names(taskmap[0])
    colmap  = [] 
    for i,name in enumerate(colnames):
        colmap.append((name,i))
    return(taskmap,colnames,colmap)

if __name__ == "__main__":
    pass