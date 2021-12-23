"""Module for Spatial layout classes
"""
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
import os
from scipy.linalg import toeplitz

class SpatialLayout():
    """Spatial layout help class for
    making full models
    """
    def __init__(self,P):
        self.P = P
    pass

class SpatialChain(SpatialLayout):
    """
        Linear arrangement of Nodes
    """
    def __init__(self,P=5):
        self.P=P
        a=np.zeros(P)
        a[1]=1
        self.W = toeplitz(a)


class SpatialGrid(SpatialLayout):
    """Rectangular grid Layout
    """
    def __init__(self,width=5,height=5):
        self.width = width
        self.height = height
        self.dim = (height,width)
        super().__init__(height*width)
        # Get the grid and neighbourhood relationship
        self.W = self.define_grid()

    def define_grid(self):
        """
        Makes a connectivity matrix and a mappin matrix for a simple rectangular grid
        """
        XX, YY = np.meshgrid(range(self.width),range(self.height))
        self.xx = XX.reshape(-1)
        self.yy = YY.reshape(-1)
        self.Dist = np.sqrt((self.xx - self.xx.reshape((-1,1)))**2 + (self.yy - self.yy.reshape((-1,1)))**2)
        W = np.double(self.Dist==1) # Nearest neighbour connectivity
        return W

    def plot_maps(self,Y,cmap='tab20',vmin=0,vmax=19,grid=None,offset=1):
        """Plots a set of map samples as an image grid
            Parameters:
                Y (nd-array): NxP data array of
                cmap (str): matplotlib color map
                vmin (double): Minimal scaling (0)
                vmax (double): Maximal scaling (19)
                grid (tuple): rows and columns of subplot grid
                offset (int): Start with subplot number (1)
        """
        if (Y.ndim == 1):
            if (grid is None):
                ax = plt.imshow(Y.reshape(self.dim),cmap=cmap,interpolation='nearest',vmin=vmin,vmax=vmax)
                ax.axes.yaxis.set_visible(False)
                ax.axes.xaxis.set_visible(False)
                return
            else:
                Y=Y.reshape(1,-1)
        N,P = Y.shape
        if grid is None:
            grid = np.zeros((2,),np.int32)
            grid[0] = np.ceil(np.sqrt(N))
            grid[1] = np.ceil(N/grid[0])
        for n in range(N):
            ax = plt.subplot(grid[0],grid[1],n+offset)
            ax.imshow(Y[n,:].reshape(self.dim),cmap=cmap,interpolation='nearest',vmin=vmin,vmax=vmax)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)


baseDir = '/Users/jdiedrichsen/Data/fs_LR_32'
hemN = ['L','R']
hem_name = ['CortexLeft','CortexRight']


class SpatialCortex(SpatialLayout):
    """
        Potts models on a single hemisphere on an isocahedron Grid
    """
    def __init__(self,K=3,hem=[0], roi_name='Icosahedron-162'):
        self.flatsurf = []
        self.inflsurf = []
        self.roi_label = []
        self.hem = hem
        self.roi_name = roi_name
        self.P = 0
        vertex = []
        for i,h in enumerate(hem):
            flatname = os.path.join(baseDir,'fs_LR.32k.' + hemN[h] + '.flat.surf.gii')
            inflname = os.path.join(baseDir,'fs_LR.32k.' + hemN[h] + '.inflated.surf.gii')
            labname = os.path.join(baseDir,roi_name + '.32k.' + hemN[h] + '.label.gii')
            sphere_name = os.path.join(baseDir,'fs_LR.32k.' + hemN[h] + '.sphere.surf.gii')

            # Get the inflate and flat surfaces and stor for later use
            self.flatsurf.append(nb.load(flatname))
            self.inflsurf.append(nb.load(inflname))

            # Get the labels and append
            roi_gifti = nb.load(labname)
            L = roi_gifti.agg_data()
            num_roi = L.max()
            L[L>0]=L[L>0]+self.P # Add the number of parcels from other hemisphere - but not to zero
            self.roi_label.append(L)
            self.P = self.P + num_roi

            # Get the vertices for the sphere for distance matrix
            sphere = nb.load(sphere_name)
            vertex.append(sphere.darrays[0].data)
            vertex[i][:,0] = vertex[i][:,0]+(h*2-1)*500

        self.coord = np.zeros((self.P,3))
        for i,h in enumerate(hem):
            for p in np.unique(self.roi_label[i]):
                if p > 0:
                    self.coord[p-1,:]= vertex[i][self.roi_label[i]==p,:].mean(axis=0)

        self.Dist = eucl_distance(self.coord)
        thresh = 1
        self.W = np.logical_and(self.Dist>0,self.Dist< thresh)
        while np.all(self.W.sum(axis=1)<=6):
            thresh = thresh+1
            self.W = np.logical_and(self.Dist>0,self.Dist< thresh)
        thresh = thresh-1
        W = np.logical_and(self.Dist>0,self.Dist< thresh)
        super().__init__(W,K)

    def map_data(self,data,out_value=0):
        """
            Args:
                data (np-arrray): 1d-array
                out_value = 0(default) or np.nan
            Returns:
                List of 1-d np-arrays (per hemisphere)
        """
        data = np.insert(data,0,out_value)
        mapped_data = []
        for i,h in enumerate(self.hem):
            mapped_data.append(data[self.roi_label[i]])
        return mapped_data

    def plot_map(self,data,cmap='tab20',vmin=0,vmax=19,grid=None):
        """
        """
        d = self.map_data(data)
        coords = self.inflsurf[0].agg_data('pointset')
        faces = self.inflsurf[0].agg_data('triangle')
        view = plotting.view_surf([coords,faces],d[0],
                cmap=cmap,vmin=vmin,vmax=vmax,symmetric_cmap = False,
                colorbar=False)
        # plotting.plot_surf([coords,faces],data[0],darkness=0.3,hemi='left')
        return view

def eucl_distance(coord):
    """
    Calculates euclediand distances over some cooordinates
    Args:
        coord (ndarray)
            Nx3 array of x,y,z coordinates
    Returns:
        dist (ndarray)
            NxN array pf distances
    """
    num_points = coord.shape[0]
    D = np.zeros((num_points,num_points))
    for i in range(3):
        D = D + (coord[:,i].reshape(-1,1)-coord[:,i])**2
    return np.sqrt(D)

