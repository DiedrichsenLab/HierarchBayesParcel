# Example Models
import numpy as np

def get_grid(width,height):
    """
        Returns a connectivity matrix and a mappin matrix for a simple rectangular grid
    """
    XX, YY = np.meshgrid(range(width),range(height))
    xx = XX.reshape(-1)
    yy = YY.reshape(-1)
    D = (xx - xx.reshape((-1,1)))**2 + (yy - yy.reshape((-1,1)))**2
    W = np.double(D==1)
    return xx,yy,D,W

class Ising:
    def __init__(self,width=5,height=5):
        self.xx, self.yy, self.D, self.W = get_grid(width,height)
        self.N = self.xx.shape[0]
        self.b = np.random.normal(0,1,(self.N,))



if __name__ == '__main__':
    B =Ising(4,4)
    pass
