#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Making random parcellation given a specific number

Created on 2/7/2025 at 3:25 PM
Author: dzhi
"""
import numpy as np
import nibabel as nb
from pathlib import Path
from HierarchBayesParcel.util import make_random_parcellation
import Functional_Fusion.atlas_map as am
import nitools as nt

base_dir = '/data/tge/Tian/UKBB_full/imaging'
if not Path(base_dir).exists():
    base_dir = '/home/dzhi/eris_mount/Tian/UKBB_full/imaging'
if not Path(base_dir).exists():
    raise NameError('Could not find base_dir')

atlas_dir = base_dir + '/Atlases'


if __name__ == '__main__':
    # Define the atlas to generate random parcellation
    atlas, _ = am.get_atlas('fs32k', atlas_dir)
    # Load surface and mask files
    surf_file_L = atlas_dir + '/tpl-fs32k/tpl-fs32k_hemi-L_sphere.surf.gii'
    mask_file_L = atlas_dir + '/tpl-fs32k/tpl-fs32k_hemi-L_mask.label.gii'
    surf_file_R = atlas_dir + '/tpl-fs32k/tpl-fs32k_hemi-R_sphere.surf.gii'
    mask_file_R = atlas_dir + '/tpl-fs32k/tpl-fs32k_hemi-R_mask.label.gii'

    # Generate 100 random parcellation given the resolution
    K = 15
    num_parcellation = 50
    rand_par = []
    for i in range(num_parcellation):
        this_left_par = make_random_parcellation(K, surf_file_L, mask_file_L)
        this_right_par = make_random_parcellation(K, surf_file_R, mask_file_R)
        this_par = np.concatenate((this_left_par, this_right_par))
        this_par = this_par[np.concatenate(atlas.vertex_mask)]
        rand_par.append(this_par)

    rand_par = np.stack(rand_par).T
    img = nt.make_label_cifti(rand_par, atlas.get_brain_model_axis(),
                              column_names=[f'parcellation_{i}' for i in range(num_parcellation)], 
                              label_names=None, label_RGBA=None)
    nb.save(img, f'random_parcellation_{num_parcellation}.dlabel.nii')