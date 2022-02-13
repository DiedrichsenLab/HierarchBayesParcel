"""Binary sparse coding model
    with variational learner
"""
import numpy as np

def sigm(x):
    return 1/(1+np.exp(-x))

class BinarySparseCode:
    def __init__(self,N=5,M=4,beta=1):
        self.N = N
        self.M = M
        self.beta = beta

    def random_init(self,seed):
        self.rng=np.random.default_rng(seed)
        self.b=self.rng.normal(0,1,(self.M,)) # Bias terms
        self.W = self.rng.normal(0,1,(self.M,self.N))

    def generate_data(self,num=1):
        h = (self.rng.random((num,self.M))<self.b).astype(int)
        y = h @ self.W + self.rng.normal(0,np.sqrt(1/self.beta),size=(num,self.N))
        return y ,h

    def variational_z(self, y,iter = 20):
        """Apply set point equations to get <h>
        """
        ind = np.arange(self.M)
        z = np.zeros((iter+1,self.M))
        h = np.ones((self.M,))*0.5
        for i in range(iter):
            for m in range(self.M):
                r = y - self.W[ind!=i,:].T @ h[ind!=i]
                z[i+1,m] = self.b[m] + r.T * self.beta @ self.W[m,:].T - 0.5 * self.W[m,:]* self.beta @ self.W[m,:].T
                h[m] = sigm(z[i+1,m])
        return z[-1,:]

def run_sim():
    N = 10
    M = BinarySparseCode(5,4,1.0)
    M.random_init(1)
    y,h = M.generate_data(N)
    z = np.zeros((N,4))
    for n in range(y.shape[0]):
        z[n,:] = M.variational_z(y[n,:])
    pass

if __name__ == '__main__':
    run_sim()