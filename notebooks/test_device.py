# Test convergence of EM for a multi-model
# general import packages
import numpy as np
import torch as pt
import time

def matrix_multiplication(dev,numformat):
    A=pt.normal(0, 1, (300, 300),device=dev, dtype=numformat)
    B=pt.normal(0, 1, (300, 300),device=dev, dtype=numformat)
    C=pt.matmul(A,B)
    return C


def test_mps_device():
    if not pt.backends.mps.is_available():
        print("MPS is not available")
        return
    dev_mps=pt.device('mps')
    dev_cpu=pt.device('cpu')
    st=time.time()
    C=matrix_multiplication(dev_cpu,pt.float32)
    en=time.time()
    print(st-en)

if __name__ == '__main__':
    test_mps_device()
