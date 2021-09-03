import os # to handle path information
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn 


def plotscatter(X,i,j,k):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:,i],X[:,j],X[:,k])
    ax.set_xlabel(f'd{i}')
    ax.set_ylabel(f'd{j}')
    ax.set_zlabel(f'd{k}')

if __name__=="__main__":
    X = np.random.normal(0,1,(30,5))
    plotscatter(X,0,1,2)
    pass