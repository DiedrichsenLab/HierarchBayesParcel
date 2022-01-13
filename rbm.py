import torch as pt
import numpy as np
import spatial as sp
import matplotlib.pyplot as plt

class RBM():
    def __init__(self, nh, nv):
        self.nh = nh
        self.nv = nv
        self.W = pt.randn(nh,nv)
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

    def sample(self,num_subj,iter=5):
        """Draw new subjects from the model

        Args:
            num_subj (int): Number of subjects
            iter (int): Number of iterations until burn in
        """
        v = pt.empty(num_subj,self.nv).uniform_(0,1)
        v = v>0.5
        v = v.type(pt.float)
        for i in range (iter):
            _,h = self.sample_h(v)
            _,v = self.sample_v(h)
        return v,h

    def epos(self, v):
        self.epos_v = v
        wv = pt.mm(v, self.W.t())
        activation = wv + self.bh
        self.epos_Eh = pt.sigmoid(activation)
        return self.epos_Eh

    def eneg_CDk(self,v,iter = 1,ph=None):
        for i in range(iter):
            _,h = self.sample_h(v)
            _,v = self.sample_v(h)
        self.eneg_Eh ,_ = self.sample_h(v)
        self.eneg_v = v
        return self.eneg_v,self.eneg_Eh

    def eneg_pCD(self,num_chains=None,iter=3):
        if (self.eneg_v is None) or (self.eneg_v.shape[0]!=num_chains):
            v = pt.empty(num_chains,self.nv).uniform_(0,1)
            v = v>0.5
        else:
            v= self.eneg_v
        for i in range(iter):
            _,h = self.sample_h(v)
            _,v = self.sample_v(h)
        self.eneg_Eh ,_ = self.sample_h(v)
        self.eneg_v = v
        return self.eneg_v,self.eneg_Eh

    def Mstep(self,alpha=0.8):
        N = self.epos_Eh.shape[0]
        M = self.eneg_Eh.shape[0]
        self.W += alpha * (pt.mm(self.epos_Eh.t(),self.epos_v) - N / M * pt.mm(self.eneg_Eh.t(),self.eneg_v))
        self.bv += alpha * pt.sum((self.epos_v - N / M * self.eneg_v), 0)
        self.bh += alpha * pt.sum((self.epos_Eh - N / M * self.eneg_Eh), 0)

    def evaluate_test(self,v,hidden,lossfcn='abserr'):
        wh = pt.mm(hidden, self.W)
        activation = wh + self.bv
        p_v = pt.sigmoid(activation)
        if lossfcn=='abserr':
            loss = pt.sum(pt.abs(v-p_v))
        elif lossfcn=='loglik':
            loss = pt.sum(v * pt.log(p_v) + (1-v) * pt.log(1-p_v))
        return loss

    def evaluate_completion(self,v,part,lossfcn='abserr'):
        num_part = part.max()+1
        loss = pt.zeros(self.nv)
        for k in range(num_part):
            ind = part==k
            v0 = pt.clone(v)
            v0[:,ind] = 0.5 # Agnostic input
            p_h = pt.sigmoid(pt.mm(v0, self.W.t()) + self.bh)
            # sample_h = pt.bernoulli(p_h)
            p_v = pt.sigmoid(pt.mm(p_h, self.W) + self.bv)
            # sample_v = pt.bernoulli(p_v_given_h)
            if lossfcn=='abserr':
                loss[ind] = pt.sum(pt.abs(v[:,ind] - p_v[:,ind]),0)
            elif lossfcn=='loglik':
                loss[ind] = pt.sum(v[:,ind] * pt.log(p_v[:,ind]) + (1-v[:,ind]) * pt.log(1-p_v[:,ind]),0)
        return pt.sum(loss)

    def evaluate_baseline(self,v,lossfcn='abserr'):
        p_v =pt.mean(v) # Overall mean 
        if lossfcn=='abserr':
            loss = pt.sum(pt.abs(v-p_v))
        elif lossfcn=='loglik':
            loss = pt.sum(v * pt.log(p_v) + (1-v) * pt.log(1-p_v))
        return loss

class mRBM():
    """multinomial (categorial) restricted Boltzman machine 
    for learning of brain parcellations
    Visible nodes: 
        The visible (most peripheral nodes) are 
        categorical with K possible categories. 
        There are three different representations: 
        a) N x nv: integers between 0 and K-1
        b) N x K x nv : indicator variables or probabilities 
        c) N x (K * nv):  Vectorized version of b- with all nodes of category 1 first, etc, 
        If not otherwise noted, we will use presentation c)
    Hidden nodes: 
        In this version we will use binary hidden nodes - so to get the same capacity as a mmRBM, one would need to set the number of hidden nodes to nh
    """
    def __init__(self, nh, nv, K):
        self.nh = nh
        self.nv = nv
        self.K = K
        self.W = pt.randn(nv*K,nh)
        self.bh = pt.randn(nh)
        self.bv = pt.randn(nv*K)
    
    def expand_mn(self,v):
        """Expands a N x nv multinomial vector 
        to an N x K x nv tensor of indictor variables
        Args:
            v (2d-tensor): N x nv matrix of samples from
        Returns
            V (3d-tensor): N x K x nv matrix of indicator variables
        """
        N = v.shape[0]
        nv = v.shape[1]
        V = pt.zeros(N,self.K,nv)
        V[np.arange(N).reshape((N,1)),v,np.arange(nv)]=1
        return V

    def compress_mn(self,V):
        """Expands a N x nv multinomial vector 
        to an N x K x nv tensor of indictor variables
        Args:
            V (3d-tensor): N x K x nv matrix of indicator variables
        Returns
            v (2d-tensor): N x nv matrix of category labels
        """
        v=V.argmax(1)
        return v

    def sample_h(self, v):
        wv = pt.mm(v.reshape(v.shape[0],-1), self.W.t())
        activation = wv + self.bh
        p_h_given_v = pt.sigmoid(activation)
        sample_h = pt.bernoulli(p_h_given_v)
        return p_h_given_v, sample_h

    def sample_v(self, h):
        """ Returns a sampled v as a unpacked indicator variable
        Args: 
            h tensor: Hidden states 
        Returns:
            v: [description]
        """
        N = h.shape[0]
        wh = pt.mm(h, self.W)
        activation = wh + self.bv
        p_v = pt.softmax(activation.reshape(self.K,self.nv),1)
        r = pt.empty(N,self.nv).uniform_(0,1)
        cdf_v = p_v.cumsum(1)
        sample_v = pt.int(r < cdf_v) 
        for k in range(self.K-1):
            sample_v[:,k-1,:]-=sample_v[:,k,:]
        return p_v, sample_v

    def sample(self,num_subj,iter=5):
        """Draw new subjects from the model

        Args:
            num_subj (int): Number of subjects
            iter (int): Number of iterations until burn in
        """
        p = pt.ones(self.K)
        v = pt.multinomial(p,num_subj*self.nv,replacement=True)
        v = v.reshape(num_subj,self.nv)
        v = self.expand_mn(v)
        for i in range (iter):
            _,h = self.sample_h(v)
            _,v = self.sample_v(h)
        V = self.compress_mn(v)
        return V,h

    def epos(self, v):
        self.epos_v = v
        wv = pt.mm(v.reshape(-1), self.W.t())
        activation = wv + self.bh
        self.epos_Eh = pt.sigmoid(activation)
        return self.epos_Eh

    def eneg_CDk(self,v,iter = 1,ph=None):
        for i in range(iter):
            _,h = self.sample_h(v)
            _,v = self.sample_v(h)
        self.eneg_Eh ,_ = self.sample_h(v)
        self.eneg_v = v
        return self.eneg_v,self.eneg_Eh

    def eneg_pCD(self,num_chains=None,iter=3):
        if (self.eneg_v is None) or (self.eneg_v.shape[0]!=num_chains):
            v = pt.empty(num_chains,self.nv).uniform_(0,1)
            v = v>0.5
        else:
            v= self.eneg_v
        for i in range(iter):
            _,h = self.sample_h(v)
            _,v = self.sample_v(h)
        self.eneg_Eh ,_ = self.sample_h(v)
        self.eneg_v = v
        return self.eneg_v,self.eneg_Eh

    def Mstep(self,alpha=0.8):
        N = self.epos_Eh.shape[0]
        M = self.eneg_Eh.shape[0]
        self.W += alpha * (pt.mm(self.epos_Eh.t(),self.epos_v) - N / M * pt.mm(self.eneg_Eh.t(),self.eneg_v))
        self.bv += alpha * pt.sum((self.epos_v - N / M * self.eneg_v), 0)
        self.bh += alpha * pt.sum((self.epos_Eh - N / M * self.eneg_Eh), 0)

    def evaluate_test(self,v,hidden,lossfcn='abserr'):
        wh = pt.mm(hidden, self.W)
        activation = wh + self.bv
        p_v = pt.sigmoid(activation)
        if lossfcn=='abserr':
            loss = pt.sum(pt.abs(v-p_v))
        elif lossfcn=='loglik':
            loss = pt.sum(v * pt.log(p_v) + (1-v) * pt.log(1-p_v))
        return loss

    def evaluate_completion(self,v,part,lossfcn='abserr'):
        num_part = part.max()+1
        loss = pt.zeros(self.nv)
        for k in range(num_part):
            ind = part==k
            v0 = pt.clone(v)
            v0[:,ind] = 0.5 # Agnostic input
            p_h = pt.sigmoid(pt.mm(v0, self.W.t()) + self.bh)
            # sample_h = pt.bernoulli(p_h)
            p_v = pt.sigmoid(pt.mm(p_h, self.W) + self.bv)
            # sample_v = pt.bernoulli(p_v_given_h)
            if lossfcn=='abserr':
                loss[ind] = pt.sum(pt.abs(v[:,ind] - p_v[:,ind]),0)
            elif lossfcn=='loglik':
                loss[ind] = pt.sum(v[:,ind] * pt.log(p_v[:,ind]) + (1-v[:,ind]) * pt.log(1-p_v[:,ind]),0)
        return pt.sum(loss)

    def evaluate_baseline(self,v,lossfcn='abserr'):
        p_v =pt.mean(v) # Overall mean 
        if lossfcn=='abserr':
            loss = pt.sum(pt.abs(v-p_v))
        elif lossfcn=='loglik':
            loss = pt.sum(v * pt.log(p_v) + (1-v) * pt.log(1-p_v))
        return loss



class RBM_grid(RBM):
    def __init__(self, width = 5, type='neighbor'):
        self.grid = sp.SpatialGrid(width=width,height=width)
        P = self.grid.P
        super().__init__(P,P)
        if type=='neighbor':
            self.grid.W = 1*self.grid.W + 2*np.eye(P)
            self.W = pt.tensor(self.grid.W,dtype=pt.float32)
            self.bh = -pt.ones(P) * self.W.sum(0).mean()/2
            self.bv = -pt.ones(P) * self.W.sum(0).mean()/2
        elif type == 'random':
            pass
        else:
            raise(NameError(f'unkown type: {type}'))

class mRBM_grid(mRBM):
    def __init__(self, K=4, width = 5, nh=100, type='random'):
        self.grid = sp.SpatialGrid(width=width,height=width)
        P = self.grid.P
        super().__init__(nh,P,K)

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
    loss = 'loglik'

    width = 4
    N = 100
    rbm_t = RBM_grid(width=width,type='neighbor')
    vtrain, hidden = rbm_t.sample(N,iter=100)
    _,vtest = rbm_t.sample_v(hidden) # Get independent test data
    num_part = 4
    p=pt.ones(num_part)/num_part
    part = pt.multinomial(p,rbm_t.nv,replacement=True)
    rbm = RBM_grid(width=width,type='random')
    nb_epoch = 20
    batch_size = 50
    ll_train = np.zeros(nb_epoch)
    ll_test1 = np.zeros(nb_epoch)
    ll_test2 = np.zeros(nb_epoch)

    for epoch in range(nb_epoch):

        # Get test error
        Eh = rbm.epos(vtrain)
        ll_train[epoch] = rbm.evaluate_test(vtrain,Eh,lossfcn=loss)
        ll_test1[epoch]= rbm.evaluate_test(vtest,Eh,lossfcn=loss)
        ll_test2[epoch]= rbm.evaluate_completion(vtest,part,lossfcn=loss)
        print(f'epoch {epoch:2d} Train: {ll_train[epoch]:.1f}, Test1: {ll_test1[epoch]:.1f}, Test2: {ll_test2[epoch]:.1f}')

        for b in range(0,N-batch_size,batch_size):
            ind = range(b,b+batch_size)
            rbm.epos(vtrain[ind,:])
            v,Eh=rbm.eneg_CDk(vtrain[ind,:],iter=1)
            rbm.Mstep(alpha=0.01)

    # Get true log likelihood
    Eh = rbm_t.epos(vtrain)
    ll_true_train = rbm_t.evaluate_test(vtest,Eh,lossfcn=loss)
    ll_true1 = rbm_t.evaluate_test(vtest,Eh,lossfcn=loss)
    ll_true2 = rbm_t.evaluate_completion(vtest,part,lossfcn=loss)
    ll_base_train = rbm_t.evaluate_baseline(vtrain,lossfcn=loss)
    ll_base_test = rbm_t.evaluate_baseline(vtest,lossfcn=loss)

    t=np.arange(0,nb_epoch)
    plt.plot(t,ll_train,'k--',label='training')
    plt.plot(t,ll_test1,'r',label='test1')
    plt.plot(t,ll_test2,'r:',label='test2')
    plt.axhline(y=ll_true1,color='r',ls='-',label='true test1')
    plt.axhline(y=ll_true2,color='r',ls=':',label='true test2')
    plt.axhline(y=ll_true_train,color='k',ls='--',label='true train')
    plt.axhline(y=ll_base_train,color='b',ls='--',label='baseline train')
    plt.axhline(y=ll_base_test,color='b',ls=':',label='baseline test')
    plt.legend()
    pass

def train_mRBM():
    # Make true hidden data set
    loss = 'abserr'

    width = 4
    N = 100
    rbm_t = mRBM_grid(K=4,nh=40,width=width,type='random')
    vtrain, hidden = rbm_t.sample(N,iter=100)
    _,vtest = rbm_t.sample_v(hidden) # Get independent test data
    num_part = 4
    p=pt.ones(num_part)/num_part
    part = pt.multinomial(p,rbm_t.nv,replacement=True)
    rbm = RBM_grid(width=width,type='random')
    nb_epoch = 20
    batch_size = 50
    ll_train = np.zeros(nb_epoch)
    ll_test1 = np.zeros(nb_epoch)
    ll_test2 = np.zeros(nb_epoch)

    for epoch in range(nb_epoch):

        # Get test error
        Eh = rbm.epos(vtrain)
        ll_train[epoch] = rbm.evaluate_test(vtrain,Eh,lossfcn=loss)
        ll_test1[epoch]= rbm.evaluate_test(vtest,Eh,lossfcn=loss)
        ll_test2[epoch]= rbm.evaluate_completion(vtest,part,lossfcn=loss)
        print(f'epoch {epoch:2d} Train: {ll_train[epoch]:.1f}, Test1: {ll_test1[epoch]:.1f}, Test2: {ll_test2[epoch]:.1f}')

        for b in range(0,N-batch_size,batch_size):
            ind = range(b,b+batch_size)
            rbm.epos(vtrain[ind,:])
            v,Eh=rbm.eneg_CDk(vtrain[ind,:],iter=1)
            rbm.Mstep(alpha=0.01)

    # Get true log likelihood
    Eh = rbm_t.epos(vtrain)
    ll_true_train = rbm_t.evaluate_test(vtest,Eh,lossfcn=loss)
    ll_true1 = rbm_t.evaluate_test(vtest,Eh,lossfcn=loss)
    ll_true2 = rbm_t.evaluate_completion(vtest,part,lossfcn=loss)
    ll_base_train = rbm_t.evaluate_baseline(vtrain,lossfcn=loss)
    ll_base_test = rbm_t.evaluate_baseline(vtest,lossfcn=loss)

    t=np.arange(0,nb_epoch)
    plt.plot(t,ll_train,'k--',label='training')
    plt.plot(t,ll_test1,'r',label='test1')
    plt.plot(t,ll_test2,'r:',label='test2')
    plt.axhline(y=ll_true1,color='r',ls='-',label='true test1')
    plt.axhline(y=ll_true2,color='r',ls=':',label='true test2')
    plt.axhline(y=ll_true_train,color='k',ls='--',label='true train')
    plt.axhline(y=ll_base_train,color='b',ls='--',label='baseline train')
    plt.axhline(y=ll_base_test,color='b',ls=':',label='baseline test')
    plt.legend()
    pass


if __name__ == '__main__':
    train_mRBM()