#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple Speed test of numpy vs torch

Created on 11/15/2022 at 1:31 PM
Author: dzhi
"""
import numpy as np
import torch as pt
import time

pt.set_default_tensor_type(pt.cuda.FloatTensor
                           if pt.cuda.is_available() else
                           pt.FloatTensor)

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
    N = 3
    x = pt.rand(200, 1000, 1000)  # Is FloatTensor
    y = pt.rand(200, 1000, 1000)
    tic = time.perf_counter()
    z = mul_torch(x, y, N)
    toc = time.perf_counter()
    print(f"Time of torch multiplication {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    z = mul_torch(x, y, N)
    toc = time.perf_counter()
    print(f"Time of torch multiplication {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    x_ = x.data.cpu()
    y_ = y.data.cpu()
    z_ = mul_numpy(x_, y_, N)
    toc = time.perf_counter()
    print(f"Time of torch multiplication {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    x_ = x.data.cpu().numpy()
    y_ = y.data.cpu().numpy()
    z_ = mul_numpy(x_, y_, N)
    toc = time.perf_counter()
    print(f"Time of torch multiplication {toc - tic:0.4f} seconds")

    pass