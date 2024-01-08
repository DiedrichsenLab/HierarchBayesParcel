#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple Speed test of M1 chip in Mac OSX
Follow install instructions: 
https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c
OR liekly better: 
https://towardsdatascience.com/gpu-acceleration-comes-to-pytorch-on-m1-macs-195c399efcc1

"""
import numpy as np
import torch as pt
import time

def use_gpu_device():
    if pt.cuda.is_available():
        # if the CUDA available machine
        device = pt.device('cuda')
        pt.set_default_device(device)
        pt.set_default_tensor_type(pt.cuda.FloatTensor)
    elif pt.backends.mps.is_available():
        # if Apple silicon M1/M2 chip available
        device = pt.device('mps')
        pt.set_default_device(device)
        pt.set_default_tensor_type(pt.FloatTensor)
    else:
        # if not - use CPU
        device = pt.device('cpu')
        pt.set_default_device(device)
        pt.set_default_tensor_type(pt.FloatTensor)

    print("The current active device is: " + device.type)


def mul_torch(x, y, n):
    z = pt.mul(x, y) # .cuda()
    for i in range(n-1):
        z += pt.mul(x, y)
    return z

def mul_numpy(x, y, n):
    z = np.multiply(x, y)
    for i in range(n-1):
        z += np.multiply(x, y)
    return z

if __name__ == '__main__':
    import pickle

    with open('E:\pre_trained models\sym_Wm_space-MNISymC3_K-20.pickle', 'rb') as file:
        models = pickle.load(file)

    # 1. move existing models (cpu) to mps device
    for m in models:
        m.move_to('mps')
    print(models[0].arrange.logpi.device)

    # 2. move mps tensors back to cpu
    for m in models:
        m.move_to('cpu')
    print(models[0].arrange.logpi.device)

    # 3. Speed test: mps vs cpu
    N = 100
    x = pt.rand((200, 1000, 1000))  # Is FloatTensor
    y = pt.rand((200, 1000, 1000))
    tic = time.perf_counter()
    z = mul_torch(x, y, N)
    toc = time.perf_counter()
    print(f"Time of torch multiplication {toc - tic:0.4f} seconds")

    use_gpu_device()
    u = pt.rand((200, 1000, 1000))  # Is FloatTensor
    v = pt.rand((200, 1000, 1000))
    tic = time.perf_counter()
    z = mul_torch(u, v, N)
    toc = time.perf_counter()
    print(f"Time of torch multiplication {toc - tic:0.4f} seconds")

    # tic = time.perf_counter()
    # x_ = x.data.cpu()
    # y_ = y.data.cpu()
    # z_ = mul_torch(x_, y_, N)
    # toc = time.perf_counter()
    # print(f"Time of torch multiplication {toc - tic:0.4f} seconds")
    #
    # tic = time.perf_counter()
    # x_ = x.data.cpu().numpy()
    # y_ = y.data.cpu().numpy()
    # z_ = mul_numpy(x_, y_, N)
    # toc = time.perf_counter()
    # print(f"Time of torch multiplication {toc - tic:0.4f} seconds")
    #
    # pass