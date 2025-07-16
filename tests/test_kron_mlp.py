import torch
import pytest
from suq.kron_suq_mlp import forward_aW_kron

################## for loop function for sanity check ##################
def forward_aW_kron_loop(a_mean, a_cov, weight, bias, A, B, b_cov):
    """
    Literal translation of Eq.(3) with Python loops.
    Shapes: see above.
    """
    N, D_in = a_mean.shape
    D_out = weight.shape[0]

    # mean
    h_mean = a_mean @ weight.T + bias

    # start with bias covariance (broadcast across batch)
    h_cov = torch.zeros((N, D_out, D_out))

    # add the three summed terms
    for n in range(N):
        for k in range(D_out):
            for l in range(D_out):
                h_cov[n, k, l] += b_cov[k, l]
                for i in range(D_in):
                    for j in range(D_in):
                        h_cov[n,k,l] += a_mean[n,i] * a_mean[n,j] * B[k,l] * A[i,j]      # term 1
                        h_cov[n,k,l] += weight[k,i] * weight[l,j] * a_cov[n,i,j]          # term 2
                        h_cov[n,k,l] += a_cov[n,i,j] * B[k,l] * A[i,j]                     # term 3
    
    return h_mean, h_cov

################## tests ##################
def test_forward_aW_kron():
    torch.manual_seed(42)
    N, D_in, D_out = 32, 15, 5

    a_mean = torch.rand([N, D_in])
    a_cov = torch.rand([N, D_in, D_in])
    weight = torch.rand([D_out, D_in])
    bias = torch.rand([D_out,])
    w_kfacs_A = torch.rand((D_in, D_in))
    w_kfacs_B = torch.rand((D_out, D_out))
    b_cov = torch.rand((D_out, D_out))

    _, h_cov = forward_aW_kron(a_mean, a_cov, weight, bias, w_kfacs_A, w_kfacs_B, b_cov)
    _, loop_h_cov = forward_aW_kron_loop(a_mean, a_cov, weight, bias, w_kfacs_A, w_kfacs_B, b_cov)

    torch.testing.assert_close(h_cov, loop_h_cov)
