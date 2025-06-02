import torch
import pytest
from suq.diag_suq_mlp import forward_aW_diag, forward_activation_implicit_diag

################## for loop function for sanity check ##################
def forward_aW_diag_loop(a_mean, a_var, weight, bias, w_var, b_var):
    
    """
    a_mean: [N, D_in] mean(a)
    a_var: [N, D_in] a_var[i] = var(a_i)
    weight: [D_out, D_in] W
    bias: [D_out, ] b
    b_var: [D_out, ] b_var[k]: var(b_k)
    w_var: [D_out, D_in] w_cov[k][i]: var(w_ki)
    """

    D_out, D_in = weight.shape
    N = a_mean.shape[0]
        
    h_mean = torch.zeros((N, D_out))
    h_var = torch.zeros((N, D_out))
    
    for n in range(N):
        for k in range(D_out):
            for i in range(D_in):
                h_mean[n][k] += a_mean[n][i] * weight[k][i]
            h_mean[n][k] += bias[k]
            
    
    for n in range(N):
        for k in range(D_out):
            for i in range(D_in):
                h_var[n][k] += a_mean[n][i]**2 * w_var[k][i] + weight[k][i]**2 * a_var[n][i] + a_var[n][i] * w_var[k][i]
            h_var[n][k] += b_var[k]
    
    return h_mean, h_var

def sigmoid_derivative(x):
    return torch.sigmoid(x) * (1. - torch.sigmoid(x))

def relu_derivative(x):
    return (x > 0).float()

def tanh_derivative(x):
    return 1.0 - torch.tanh(x) ** 2

def forward_activation_implicit_diag_loop(activation_function, activation_derivative, h_mean, h_var):
    
    N, D_in = h_mean.shape
    
    a_mean = torch.zeros((N, D_in))
    a_var = torch.zeros((N, D_in))
    
    for k in range(N):
        for i in range(D_in):
            a_mean[k][i] = activation_function(h_mean[k][i])
            a_var[k][i] = activation_derivative(h_mean[k][i]) ** 2 * h_var[k][i]
            
    return a_mean, a_var

################## tests ##################
def test_forward_aW_diag():
    torch.manual_seed(42)
    N, D_in, D_out = 32, 100, 50

    a_mean = torch.rand([N, D_in])
    a_var = torch.rand([N, D_in])
    weight = torch.rand([D_out, D_in])
    bias = torch.rand([D_out,])
    w_var = torch.rand((D_out, D_in))
    b_var = torch.rand((D_out, ))
    h_mean = torch.rand([N, D_in])
    h_var = torch.rand([N, D_in])

    h_mean, h_var = forward_aW_diag(a_mean, a_var, weight, bias, w_var, b_var)
    loop_h_mean, loop_h_var = forward_aW_diag_loop(a_mean, a_var, weight, bias, w_var, b_var)

    torch.testing.assert_close(h_mean, loop_h_mean)
    torch.testing.assert_close(h_var, loop_h_var)

@pytest.mark.parametrize("activation, derivative", [
    (torch.sigmoid, sigmoid_derivative),
    (torch.relu, relu_derivative),
    (torch.tanh, tanh_derivative),
])

def test_forward_activation_implicit_diag(activation, derivative):
    torch.manual_seed(42)
    N, D_in = 32, 50
    h_mean = torch.rand([N, D_in])
    h_var = torch.rand([N, D_in])

    a_mean, a_var = forward_activation_implicit_diag(activation, h_mean, h_var)
    loop_a_mean, loop_a_var = forward_activation_implicit_diag_loop(activation, derivative, h_mean, h_var)

    torch.testing.assert_close(a_mean, loop_a_mean)
    torch.testing.assert_close(a_var, loop_a_var)
