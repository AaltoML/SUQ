import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils import parameters_to_vector
import numpy as np
import copy

from suq.base_suq import SUQ_Base

def forward_aW_diag(a_mean, a_var, weight, bias, w_var, b_var):
    """
    compute mean and covariance of h = a @ W^T + b when posterior has diag covariance
    
    ----- Input -----
    a_mean: [N, D_in] mean(a)
    a_var: [N, D_in] a_var[i] = var(a_i)
    weight: [D_out, D_in] W
    bias: [D_out, ] b
    b_var: [D_out, ] b_var[k]: var(b_k)
    w_var: [D_out, D_in] w_cov[k][i]: var(w_ki)
    ----- Output -----
    h_mean: [N, D_out]
    h_var: [N, D_out] h_var[k] = var(h_k)
    """
    
    # calculate mean(h)
    h_mean = F.linear(a_mean, weight, bias)
    
    # calculate var(h)
    weight_mean2_var_sum = weight ** 2 + w_var # [D_out, D_in]
    h_var = a_mean **2 @ w_var.T + a_var @ weight_mean2_var_sum.T + b_var
    
    return h_mean, h_var


def forward_activation_implicit_diag(activation_func, h_mean, h_var):
    """
    given h ~ N(h_mean, h_cov), g(·), where h_cov is a diagonal matrix,
    approximate the distribution of a = g(h) as 
    a ~ N(g(h_mean), g'(h_mean)^T h_var g'(h_mean))
    
    input
    activation_func: g(·)
    h_mean: [N, D]
    h_var: [N, D], h_var[i] = var(h_i)
    
    output
    a_mean: [N, D]
    a_var: [N, D]
    """

    h_mean_grad = h_mean.detach().clone().requires_grad_()
    
    a_mean = activation_func(h_mean_grad)
    a_mean.retain_grad()
    a_mean.backward(torch.ones_like(a_mean)) #[N, D]
    
    nabla = h_mean_grad.grad #[N, D]
    a_var = nabla ** 2 * h_var
    
    return a_mean.detach(), a_var

def forward_batch_norm_diag(h_var, bn_weight, bn_running_var, bn_eps):
    """
    Pass a distribution with diagonal covariance through BatchNorm layer
    
    Input
        h_mean: mean of input distribution [B, D]
        h_var: variance of input distribution [B, D]
        bn_weight: batch norm scale factor [D, ]
        bn_running_var: batch norm running variance [D, ]
        bn_eps: batch norm eps
    
    Output
        output_var [B, T, D]
    """

    scale_factor = (1 / (bn_running_var.reshape(1, -1) + bn_eps)) * bn_weight.reshape(1, -1) **2 # [B, D]
    output_var = scale_factor * h_var # [B, D]
    
    return output_var

class SUQ_Linear_Diag(nn.Module):

    def __init__(self, org_linear, w_var, b_var):
        super().__init__()
        
        self.weight = org_linear.weight.data
        self.bias = org_linear.bias.data
        self.w_var = w_var
        self.b_var = b_var
    
    def forward(self, a_mean, a_var): 
        
        if a_var == None:
            a_var = torch.zeros_like(a_mean).to(a_mean.device)
            
        h_mean, h_var = forward_aW_diag(a_mean, a_var, self.weight, self.bias, self.w_var, self.b_var)
        
        return h_mean, h_var

class SUQ_Activation_Diag(nn.Module):
    
    def __init__(self, afun):        
        super().__init__()
        self.afun = afun
    
    def forward(self, h_mean, h_var):
        a_mean, a_var = forward_activation_implicit_diag(self.afun, h_mean, h_var)
        return a_mean, a_var

class SUQ_BatchNorm_Diag(nn.Module):
    
    def __init__(self, BatchNorm):
        super().__init__()
        
        self.BatchNorm = BatchNorm
    
    def forward(self, x_mean, x_var):
        
        with torch.no_grad():
        
            out_mean = self.BatchNorm.forward(x_mean)
            out_var = forward_batch_norm_diag(x_mean, x_var, self.BatchNorm.weight, 1e-5)
            
        return out_mean, out_var


class SUQ_MLP_Diag(SUQ_Base):
    def __init__(self, org_model, posterior_variance, likelihood, scale_init = 1.0, sigma_noise = None):
        super().__init__(likelihood, scale_init)
        """
        only take in the model that corresponds to the latent function part (in regression case it's the whole network, in classification case remove the softmax layer)
        """
        self.sigma_noise = sigma_noise
        self.convert_model(org_model, posterior_variance)
    
    def forward_latent(self, data, out_var = None):
        out_mean = data
        
        if isinstance(self.model, nn.Sequential):
            for layer in self.model:
                out_mean, out_var = layer.forward(out_mean, out_var)
        ##TODO: other type of model            

        out_var = out_var / self.scale_factor
        
        return out_mean, out_var
    
    def forward(self, data):
        
        out_mean, out_var = self.forward_latent(data)

        if self.likelihood == 'classification':
            kappa = 1 / torch.sqrt(1. + np.pi / 8 * out_var)
            return torch.softmax(kappa * out_mean, dim=-1)

        if self.likelihood == 'regression':
            return out_mean, out_var + self.sigma_noise ** 2
    
    def convert_model(self, org_model, posterior_variance):
        
        """
        take the torch model and its corresponding posterior, convert it into Bayesian version
        """
        
        p_model = copy.deepcopy(org_model)

        loc = 0
        for n, layer in p_model.named_modules():
            if isinstance(layer, nn.Linear):
                
                D_out, D_in = layer.weight.data.shape
                num_param = torch.numel(parameters_to_vector(layer.parameters()))
                num_weight_param = D_out * D_in
                
                covariance_block = posterior_variance[loc : loc + num_param]
                
                """
                w_var: [D_out, D_in], w_var[k][i] = var(w_ki)
                b_var: [D_out, ] b_var[k]: var(b_k)
                """        
                
                b_var = torch.zeros_like(layer.bias.data).to(layer.bias.data.device)
                w_var = torch.zeros_like(layer.weight.data).to(layer.bias.data.device)

                for k in range(D_out):
                    b_var[k] = covariance_block[num_weight_param + k]
                    for i in range(D_in):
                        w_var[k][i] = covariance_block[k * D_in + i]

                new_layer = SUQ_Linear_Diag(layer, w_var, b_var)

                loc += num_param
                setattr(p_model, n, new_layer)
            
            if isinstance(layer, nn.BatchNorm1d):
                new_layer = SUQ_BatchNorm_Diag(layer)
                setattr(p_model, n, new_layer)
                
            if type(layer).__name__ in torch.nn.modules.activation.__all__:
                new_layer = SUQ_Activation_Diag(layer)
                setattr(p_model, n, new_layer)

        self.model = p_model