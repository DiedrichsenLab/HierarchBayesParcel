import torch as pt
import numpy as np
import spatial as sp
import matplotlib.pyplot as plt

class RBM():
    def __init__(self, width):
        self.grid = sp.SpatialGrid(width=width,height=width)
        nh = nv = self.grid.P
        self.W = pt.randn(nv,nh)
        self.bh = pt.randn(nh)
        self.bv = pt.randn(nv)

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

def plot_batch(v0,vk,ph0,phk,rbm):
    plt.figure()
    N=4
    for n in range(N):
        plt.subplot(4,N,n+1)
        rbm.grid.plot_maps(v0[n],vmax=1,cmap='jet')
        plt.subplot(4,N,n+N+1)
        rbm.grid.plot_maps(ph0[n],vmax=1,cmap='jet')
        plt.subplot(4,N,n+2*N+1)
        rbm.grid.plot_maps(vk[n],vmax=1,cmap='jet')
        plt.subplot(4,N,n+3*N+1)
        rbm.grid.plot_maps(phk[n],vmax=1,cmap='jet')


def train_RBM():
    # Make true hidden data set
    width = 4
    N = 200
    rbm_t = RBM(width)
    P =rbm_t.grid.P
    rbm_t.grid.W = 1*rbm_t.grid.W + 2*np.eye(P)
    rbm_t.W = pt.tensor(rbm_t.grid.W,dtype=pt.float32)
    rbm_t.bh = -pt.ones(P) * rbm_t.W.sum(0).mean()/2
    rbm_t.bv = -pt.ones(P) * rbm_t.W.sum(0).mean()/2
    ph = pt.empty(N,P).uniform_(0,1)
    h_train=pt.bernoulli(ph)
    # plt.figure()
    for k in range(6):
        # plt.subplot(2,6,k+1)
        # rbm_t.grid.plot_maps(ph[0],cmap='jet',vmin=0,vmax=1)
        pv,v_train=rbm_t.sample_v(h_train)
        # plt.subplot(2,6,k+7)
        # rbm_t.grid.plot_maps(pv[0],cmap='jet',vmin=0,vmax=1)
        ph,h_train=rbm_t.sample_h(v_train)
    pass
    # Now generate new RBM and use for fitting
    rbm = RBM(width)
    # rbm.grid.W = 2*rbm.grid.W + 4*np.eye(P)
    # rbm.W = pt.tensor(rbm.grid.W,dtype=pt.float32)
    # rbm.bh = -pt.ones(P) * rbm.W.sum(0).mean()/2
    # rbm.bv = -pt.ones(P) * rbm.W.sum(0).mean()/2

    nb_epoch = 20
    batch_size = 20
    loss_train = np.zeros(nb_epoch)

    for epoch in range(nb_epoch):
        for n in range(0, N - batch_size, batch_size):
            vk = v_train[n: n+batch_size]
            v0 = v_train[n: n+batch_size]
            ph0,_ = rbm.sample_h(v0)
            for k in range(10):
                _, hk = rbm.sample_h(vk)
                _, vk = rbm.sample_v(hk)
            phk, _ = rbm.sample_h(vk)
            rbm.train(v0, vk, ph0, phk,alpha=0.01)
            loss_train[epoch] += pt.mean(pt.abs(v0-vk))
            # plot_batch(v0,vk,ph0,phk,rbm)
            pass
        print(f"Epoch {epoch}, Loss {loss_train[epoch]:.3f}")
    # Now generate test_loss
    plt.plot(loss_train)
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