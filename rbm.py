import torch as pt
import numpy as np
import spatial as sp
import matplotlib.pyplot as plt

class RBM():
    def __init__(self, width):
        self.grid = sp.SpatialGrid(width=width,height=width)
        self.grid.W = self.grid.W + np.eye(self.grid.P)
        self.W = pt.tensor(self.grid.W,dtype=pt.float32)
        nh = nv = self.grid.P
        self.bh = pt.randn(nh)
        self.bv = pt.zeros(nv)

    def sample_h(self, v):
        wv = pt.mm(v, self.W.t())
        activation = wv + self.bh
        p_h_given_v = pt.sigmoid(activation)
        sample_h = pt.bernoulli(p_h_given_v)
        return p_h_given_v, sample_h

    def sample_v(self, h):
        wh = pt.mm(h, self.W)
        activation = wh + self.bv
        p_v_given_h = pt.sigmoid(activation)
        sample_v = pt.bernoulli(p_v_given_h)
        return p_v_given_h, sample_v

    def train(self, v0, vk, ph0, phk, alpha=0.5):
        self.W += alpha * pt.mm(ph0.t(),v0) - pt.mm(phk.t(),vk)
        self.bv += alpha * pt.sum((v0 - vk), 0)
        self.bh += alpha * pt.sum((ph0 - phk), 0)

def train_RBM():
    # Make true hidden data set
    width = 10
    N = 120
    rbm_t = RBM(width)
    P = width*width
    ph = pt.empty(N,P).uniform_(0,1)

    h_train=pt.bernoulli(ph)
    plt.figure()
    for k in range(6):
        _,v_train=rbm_t.sample_v(h_train)
        _,h_train=rbm_t.sample_v(v_train)
        plt.subplot(2,3,k+1)
        rbm_t.grid.plot_maps(v_train[0])
    pass
    # Now generate new RBM and use for fitting
    rbm = RBM(nv, nh)
    nb_epoch = 10
    batch_size = 10
    for epoch in range(1, nb_epoch+1):
        train_loss = 0
        s = 0.0
        for n in range(0, N - batch_size, batch_size):
            vk = v_train[n: n+batch_size]
            v0 = v_train[n: n+batch_size]
            ph0,_ = rbm.sample_h(v0)
            for k in range(10):
                _, hk = rbm.sample_h(vk)
                _, vk = rbm.sample_v(hk)
                vk[v0<0] = v0[v0<0]
            phk, _ = rbm.sample_h(vk)
            rbm.train(v0, vk, ph0, phk)
        train_loss += pt.mean(pt.abs(v0-vk))
        s += 1
        print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

    # Now generate test_loss
    test_loss = 0
    s = 0.
    for id_user in range(0, nb_users):
        v_input = training_set[id_user: id_user+1]
        v_target = test_set[id_user: id_user+1]
        if len(v_target(v_target>=0)):
            _, h = rbm.sample_h(v_input)
            _, v_input = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(v_target[v_target>0]-
                                          v_input[v_target>0]))
        s += 1

if __name__ == '__main__':
    train_RBM()