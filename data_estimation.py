#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 10/26/2021
The script for finding the best distribution to describe the MDTB dataset.
Given all the mean-centered data and calculated for each vertex's magnitude

This method can be take any distribution given the all subject data, By
default, our prior knowledge is that the data fits the gamma distribution
Author: DZHI
'''
import numpy as np
import scipy.stats as stats
import os
import scipy.io as spio
import nibabel as nb
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import dict_learning, sparse_encode
from sklearn.manifold import TSNE

goodsubj = [2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31]
subj_name = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11',
             's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24',
             's25', 's26', 's27', 's28', 's29', 's30', 's31']


def load_subjectData(path, hemis='L'):
    """
        Load subject data in .mat and func.gii format.
        This function can be further extended depending on user's data file type, default subject data type is func.gii.
        If type = func.gii, the file name must be subNum.hemisphere.wbeta.resolution.func.gii
                            i.g 's02.L.wbeta.32k.func.gii'
        :param path: the current subject data folder path
        :return: subject data, expect data shape [N, k]
                    N is the number of vertices
                    k is the number of task conditions
    """
    files = os.listdir(path)

    if not any(".mat" or ".func.gii" in x for x in files):
        raise Exception('Input data file type is not supported.')
    elif any(".mat" in x for x in files):
        data = spio.loadmat(os.path.join(path, "file_%s.txt" % hemis))
    else:
        '''Default data type is func.gii'''
        sub = '.%s.' % hemis
        fileName = [s for s in files if sub in s]
        mat = nb.load(os.path.join(path, fileName[0]))
        wbeta_data = [x.data for x in mat.darrays]
        wbeta = np.reshape(wbeta_data, (len(wbeta_data), len(wbeta_data[0])))
        data = wbeta.transpose()

    return data


colors = plt.cm.rainbow(np.linspace(0, 1, 24))
alpha, loc, beta = [], [], []
fig, ax = plt.subplots(1, 1)
fig1 = make_subplots(rows=4, cols=6,
                     specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
                            [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
                            [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
                            [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
                     )
for i in range(len(goodsubj)):
    path = 'data/%s' % subj_name[goodsubj[i]-1]
    data = load_subjectData(path, hemis='L')
    mean = data.mean(axis=1)
    data = data - mean[:, np.newaxis]
    mag = np.linalg.norm(data, axis=1)

    # fig1 = px.scatter_3d(x=data[:, np.random.randint(34)], y=data[:, np.random.randint(34)], z=data[:, np.random.randint(34)], opacity=0.3, template='plotly_white')
    # fig1.show()
    # fig1.add_trace(
    #     go.Scatter3d(x=data[:, np.random.randint(34)], y=data[:, np.random.randint(34)], z=data[:, np.random.randint(34)],
    #                  mode='markers', marker=dict(size=3, opacity=0.7)),
    #                  row=np.int(np.floor(i/6)+1), col=(i % 6)+1)

    tsne = TSNE(n_components=3, random_state=0)
    X_embadded = tsne.fit_transform(data[~np.isnan(data).any(axis=1), :])

    num = 10
    Uhat = np.empty((num, P, K))
    Vhat = np.empty((num, K, N))
    for i in range(num):
        # Determine random starting value
        V_init = random_V(K, N)
        U_init = sparse_encode(Y, V_init, alpha=1, algorithm='lasso_cd')
        Uhat[i, :, :], Vhat[i, :, :], errors = dict_learning(Y, alpha=1, n_components=5, method='cd',
                                                             random_state=i, positive_code=True,
                                                             code_init=U_init, dict_init=V_init)

    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(mag[~np.isnan(mag)])
    alpha = np.append(alpha, fit_alpha)
    loc = np.append(loc, fit_loc)
    beta = np.append(beta, fit_beta)

    x = np.linspace(0, 6, 100)
    y = stats.gamma.pdf(x, a=fit_alpha, loc=fit_loc, scale=fit_beta)

    ax.plot(x, y, color=colors[i])
    # plt.legend(loc='upper right')

plt.title('gamma distribution for 24 subjects of MDTB')
plt.show()
fig1.show()
print(alpha, loc, beta)

