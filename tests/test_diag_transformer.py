import torch
import pytest
from suq.diag_suq_transformer import forward_value_cov_determinstic_W, forward_value_cov_Bayesian_W, forward_QKV_cov, forward_fuse_multi_head_cov
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################## einsum function for sanity check ##################
def einsum_V_cov(Wv_mean, Wv_var, x_mean, x_var, n_h, D_v):
    
    D = n_h * D_v
    
    # Covariance of values
    Wv_mean = Wv_mean.view(n_h, D_v, D)   # (nh, hs, D)
    #Wv_var = F.softplus(Wv_var.view(n_h, D_v, D))**2 # (nh, hs, D)
    Wv_var = Wv_var.view(n_h, D_v, D)
    # (nh, hs, D), (B, T, D), (nh, hs, D) -> (B, T, nh, hs, hs)
    v_cov = torch.einsum('ijk,lmk,ink->lmijn', Wv_mean, x_var, Wv_mean)
    v_cov += torch.diag_embed(
        # (B, T, D), (nh, hs, D) -> (B, T, nh, hs)
        torch.einsum('ijk,lmk->ijlm', x_var, Wv_var) +
        # (B, T, D), (nh, hs, D) -> (B, T, nh, hs)
        torch.einsum('ijk,lmk->ijlm', x_mean**2, Wv_var)
    )
    
    return v_cov

def einsum_QKV(attention_score, v_cov):

    QKV_cov = torch.einsum('ijkl,iljmn->ijkmn', attention_score**2, v_cov)
    
    return QKV_cov

def einsum_multi_head_fuse(QKV_cov, project_W):
    B, n_h, T, D_v, _ = QKV_cov.size()
    D, _ = project_W.shape
    
    Wo_mean = project_W.T.reshape(n_h, D_v, D)
    output_var = torch.einsum('ijk,limjn,ink->lmk', Wo_mean, QKV_cov, Wo_mean)
    return output_var

################## tests ##################
def test_V_cov():
    B, T, D, n_h, D_v = 4, 20, 100, 5, 20
    torch.manual_seed(42)
    
    input_mean = torch.rand((B, T, D)).to(device)
    input_var = torch.rand((B, T, D)).to(device)
    W_v_var = torch.zeros((D, D)).to(device)
    W_v = torch.rand((D, D)).to(device)
    
    res_1 = einsum_V_cov(W_v, W_v_var, input_mean, input_var, n_h, D_v)
    res_2 = forward_value_cov_Bayesian_W(W_v, W_v_var, input_mean, input_var, n_h, D_v, False)

    torch.testing.assert_close(res_1, res_2, atol=5e-5, rtol=1e-5)

def test_Bayesian_non_Bayesian_match():
    
    B, T, D, n_h, D_v = 4, 20, 100, 5, 20
    torch.manual_seed(42)
    
    input_mean = torch.rand((B, T, D)).to(device)
    input_var = torch.rand((B, T, D)).to(device)
    W_v_var = torch.zeros((D, D)).to(device)
    W_v = torch.rand((D, D)).to(device)
    
    # Bayesian verison with zero variance should return the same as determinstic one 
    v_cov_one = forward_value_cov_Bayesian_W(W_v, W_v_var, input_mean, input_var, n_h, D_v, False)
    v_cov_two = forward_value_cov_determinstic_W(W_v, input_var, n_h, D_v, False)
    torch.testing.assert_close(v_cov_one, v_cov_two)

    v_cov_one = forward_value_cov_Bayesian_W(W_v, W_v_var, input_mean, input_var, n_h, D_v, True)
    v_cov_two = forward_value_cov_determinstic_W(W_v, input_var, n_h, D_v, True)

    torch.testing.assert_close(v_cov_one, v_cov_two, atol=5e-5, rtol=1e-5)
    
def test_QKV_cov():
    
    B, T, D, n_h, D_v = 4, 20, 100, 5, 20
    torch.manual_seed(42)
    
    attention_score = torch.rand((B, n_h, T, T)).to(device)
    v_cov = torch.rand((B, T, n_h, D_v, D_v)).to(device)
    
    res_1 = forward_QKV_cov(attention_score, v_cov)
    res_2 = einsum_QKV(attention_score, v_cov)

    torch.testing.assert_close(res_1, res_2, atol=5e-5, rtol=1e-5)

def test_fuse_multi_head_cov():
    
    B, T, D, n_h, D_v = 4, 20, 100, 5, 20
    torch.manual_seed(42)
    
    QKV_cov = torch.rand((B, n_h, T, D_v, D_v)).to(device)
    project_W = torch.rand((D, D)).to(device)
    
    res_1 = forward_fuse_multi_head_cov(QKV_cov, project_W)
    res_2 = einsum_multi_head_fuse(QKV_cov, project_W)

    torch.testing.assert_close(res_1, res_2, atol=5e-5, rtol=1e-5)

def test_diag_value_cov():
    B, T, D, n_h, D_v = 4, 20, 100, 5, 20
    torch.manual_seed(42)
    
    input_mean = torch.rand((B, T, D)).to(device)
    input_var = torch.rand((B, T, D)).to(device)
    W_v_var = torch.zeros((D, D)).to(device)
    W_v = torch.rand((D, D)).to(device)
    
    diag_res = forward_value_cov_Bayesian_W(W_v, W_v_var, input_mean, input_var, n_h, D_v, True)
    cov_res = forward_value_cov_Bayesian_W(W_v, W_v_var, input_mean, input_var, n_h, D_v, False)

    torch.testing.assert_close(diag_res, cov_res.diagonal(dim1=-2, dim2=-1), atol=5e-5, rtol=1e-5)

def test_diag_QKV_cov():

    B, T, D, n_h, D_v = 4, 20, 100, 5, 20
    torch.manual_seed(42)
    
    attention_score = torch.rand((B, n_h, T, T)).to(device)
    v_cov = torch.rand((B, T, n_h, D_v, D_v)).to(device)
    
    cov_res = forward_QKV_cov(attention_score, v_cov, False)
    diag_res = forward_QKV_cov(attention_score, v_cov.diagonal(dim1=-2, dim2=-1), True)

    torch.testing.assert_close(diag_res, cov_res.diagonal(dim1=-2, dim2=-1), atol=5e-5, rtol=1e-5)

def test_diag_fuse_multi_head_cov():
    B, T, D, n_h, D_v = 4, 20, 100, 5, 20
    torch.manual_seed(42)
    
    QKV_var = torch.rand((B, n_h, T, D_v)).to(device)
    QKV_cov = torch.diag_embed(QKV_var)
    project_W = torch.rand((D, D)).to(device)
    
    diag_res = forward_fuse_multi_head_cov(QKV_var, project_W, True)
    cov_res = forward_fuse_multi_head_cov(QKV_cov, project_W, False)
    
    torch.testing.assert_close(diag_res, cov_res, atol=5e-5, rtol=1e-5)


    

    

    
