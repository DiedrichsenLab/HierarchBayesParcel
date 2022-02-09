"""
Evaluation - implements evaluation of emission, arrangement, or full models 
Assumes that data, likelihoods, and estimates comes as NxKxP tensors
First are basic functions for save evaluation - 
Second are more complex functions that use different criteria 
""" 
import torch as pt 
import numpy as np

def u_abserr(U,uhat):
    """Absolute error on U 
    Args:
        U (tensor): Real U's 
        uhat (tensor): Estimated U's from arangement model 
    """
    return pt.mean(pt.abs(U-uhat))

def logpY(emloglik,Uhat,offset='P'):
    """Averaged log of p(Y|U)
    For save computation (to prevent underflow or overflow) 
    p(y|u) either gets a constant
    """
    if offset is None:
        pyu = pt.exp(emloglik)
    elif offset=='P':
        pyu = pt.softmax(emloglik,dim=1)
    elif type(offset) is float:
        pyu = pt.exp(emloglik-offset)
    else:
        raise(NameError('offset needs to be P,None, or floatingpoint'))
    py = pt.sum(pyu * Uhat,dim=1)
    return pt.mean(pt.log(py),dim=(0,1))

def evaluate_full_arr(data,Uhat,crit='logpY',offset='P'):
    """Evaluates an arrangement model new data set using pattern completion from partition to partition, using a leave-one-partition out crossvalidation approach.
    Args:
        data (tensor): Emission log-liklihood, Us (depends on crit)
        Uhat (tensor): Probility for each node (expected U) 
        crit (str): 'logpy','u_abserr'
    Returns:
        evaluation citerion: [description]
    """
    if type(data) is np.ndarray:
        data=pt.tensor(data,dtype=pt.get_default_dtype())
    if crit=='logpY': 
        U = pt.softmax(data,dim=1)
        emloglik = data
        return logpY(emloglik,Uhat)
    elif crit == 'u_abserr': 
        U = data
        return u_abserr(U,Uhat)


def evaluate_completion_arr(arM,data,part,crit='logpY',offset='P'):
    """Evaluates an arrangement model new data set using pattern completion from partition to partition, using a leave-one-partition out crossvalidation approach.
    Args:
        arM (ArrangementModel): [description]
        data (tensor): Emission log-liklihood or U-estimates (depends on crit)
        part (Partitions): P tensor with partition indices 
        crit (str): 'logpy','u_abserr'
    Returns:
        evaluation citerion: [description]
    """
    if type(data) is np.ndarray:
        data=pt.tensor(data,dtype=pt.get_default_dtype())
    if crit=='logpY': 
        U = pt.softmax(data,dim=1)
        emloglik = data
        if offset is None:
            pyu = pt.exp(emloglik)
        elif offset=='P':
            pyu = pt.softmax(emloglik,dim=1)
        elif type(offset) is float:
            pyu = pt.exp(emloglik-offset)
        else:
            raise(NameError('offset needs to be P,None, or floatingpoint'))
    elif crit == 'u_abserr': 
        U = data
    else:
        raise(NameError('unknown criterion'))
    N = U.shape[0]
    num_part = part.max()+1
    loss = pt.zeros(arM.P)
    for k in range(num_part):
        ind = part==k
        U0 = pt.clone(U)
        U0[:,:,ind] = 1./arM.K # Agnostic input
        Uhat,_ = arM.Estep(U0,gather_ss=False) 
        if crit=='abserr':
            loss[ind] = pt.mean(pt.abs(U[:,:,ind] - Uhat[:,:,ind]),dim=(0,1))
        elif crit=='logpY':
            py = pt.sum(pyu[:,:,ind] * Uhat[:,:,ind],dim=1)
            loss[ind] = pt.mean(pt.log(py),dim=0)
    return pt.mean(loss) # average across vertices 
