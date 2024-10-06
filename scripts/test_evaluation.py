import HierarchBayesParcel.evaluation as ev
import HierarchBayesParcel.arrangements as ar
import torch as pt

def test_cosine_error():
    """ Test different forms of the cosine error function """
    N = 7
    K = 4
    P = 1000
    n_sub = 3
    # Make a V matrix
    V = pt.randn((N, K))
    V = V / V.norm(dim=0,keepdim=True)

    # Make a U matrix
    U = pt.rand((n_sub,K,P))
    U = U / U.sum(dim=1,keepdim=True)
    Uhard = ar.expand_mn(ar.compress_mn(U),K)

    # Make different datasets
    Y_rand = pt.randn((n_sub,N, P))
    Y_hard = pt.matmul(V,Uhard)
    Y_avrg = pt.matmul(V,U)
    type = ['hard','average','expected']
    ytyp = ['random','hard','average']


    # Test 1: different simulation scenarios
    for i,Y in enumerate([Y_rand, Y_hard, Y_avrg]):
        # Test the cosine error
        for t in type:
            error = ev.coserr(Y,V,U, type=t,adjusted=False)
            print(f"{t} cos error on {ytyp[i]} data: {error}")
    pass

    # Test 2: Different sizes
    error = ev.coserr(Y_avrg,V,U[0,:,:], type='average',adjusted=False)
    print(f"cos error with 1 prediction: {error}")
    error = ev.coserr(Y_avrg[0,:,:],V,U[0,:,:], type='average',adjusted=False)
    print(f"cos error with 1 size: {error}")
    pass



if __name__ == "__main__":
    test_cosine_error()