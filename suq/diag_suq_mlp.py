import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils import parameters_to_vector
import numpy as np
import copy

from suq.base_suq import SUQ_Base

def forward_aW_diag(a_mean, a_var, weight, bias, w_var, b_var):
    """
    Compute the mean and element-wise variance of `h = a @ W^T + b` when the posterior has diagonal covariance.
    
    Args:
        a_mean (Tensor): Mean of the input `a`. Shape: `[B, D_in]`.
        a_var (Tensor): Variance of the input `a`. Shape: `[B, D_in] `
        weight (Tensor): Mean of the weights `W`. Shape: `[D_out, D_in]`
        bias (Tensor): Mean of the bias `b`. Shape: `[D_out, ]`
        b_var (Tensor): Element-wise variance of the bias `b`. Shape: `[D_out, ]`
        w_var (Tensor): Element-wise variance of the weights `W`. Shape: `[D_out, D_in]`
    
    Returns: 
        h_mean (Tensor): Mean of the pre-activations `h`. Shape: `[B, D_out]`
        h_var (Tensor): Element-wise variance of the pre-activations `h`. Shape: `[B, D_out]`
    """
    
    # calculate mean(h)
    h_mean = F.linear(a_mean, weight, bias)
    
    # calculate var(h)
    weight_mean2_var_sum = weight ** 2 + w_var # [D_out, D_in]
    h_var = a_mean **2 @ w_var.T + a_var @ weight_mean2_var_sum.T + b_var
    
    return h_mean, h_var


def forward_activation_implicit_diag(activation_func, h_mean, h_var):
    
    """
    Approximate the distribution of `a = g(h)` given `h ~ N(h_mean, h_var)`, where `h_var` 
    is the element-wise variance of pre-activation `h`.
    Uses a first-order Taylor expansion: `a ~ N(g(h_mean), g'(h_mean)^T @ h_var @ g'(h_mean))`.

    Args:
        activation_func (Callable): A PyTorch activation function `g(·)` (e.g. `nn.ReLU()`)
        h_mean (Tensor): Mean of the pre-activations `h`. Shape: `[B, D]`
        h_var (Tensor): Element-wise variance of the pre-activations `h`. Shape: `[B, D]`

    Returns:
        a_mean (Tensor): Mean of the activations `a`. Shape: `[B, D]`
        a_var (Tensor): Element-wise variance of the activations `a`. Shape: `[B, D]`
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
    Compute the output variance when a distribution `h ~ N(h_mean, h_var)`
    is passed through a BatchNorm layer.

    Args:
        h_var (Tensor): Element-wise variance of the input `h`. Shape: `[B, D]`.
        bn_weight (Tensor): Batch normalization scale factor (gamma). Shape: `[D,]`.
        bn_running_var (Tensor): Running variance used in batch normalization. Shape: `[D,]`.
        bn_eps (float): Small constant added to the denominator for numerical stability.

    Returns:
        output_var (Tensor): Element-wise variance of the output after batch normalization. Shape: `[B, D]`.
    """

    scale_factor = (1 / (bn_running_var.reshape(1, -1) + bn_eps)) * bn_weight.reshape(1, -1) **2 # [B, D]
    output_var = scale_factor * h_var # [B, D]
    
    return output_var

class SUQ_Linear_Diag(nn.Module):
    """
    Linear layer with uncertainty propagation under SUQ, with a diagonal Gaussian posterior.
    
    Wraps a standard `nn.Linear` layer and applies closed-form mean and variance propagation. See the SUQ paper for theoretical background and assumptions.

    Args:
        org_linear (nn.Linear): The original linear layer to wrap      
        w_var (Tensor): Element-wise variance of the weights `W`. Shape: `[D_out, D_in]`
        b_var (Tensor): Element-wise variance of the bias `b`. Shape: `[D_out, ]`
    """
    def __init__(self, org_linear, w_var, b_var):
        super().__init__()
        
        self.weight = org_linear.weight.data
        self.bias = org_linear.bias.data
        self.w_var = w_var
        self.b_var = b_var
    
    def forward(self, a_mean, a_var): 
        """
        Forward pass with uncertainty propagation through a SUQ linear layer.
        
        Args:
            a_mean (Tensor): Input mean. Shape: `[B, D_in]`
            a_var (Tensor): Input element-wise variance. Shape: `[B, D_in]`

        Returns:
            h_mean (Tensor): Mean of the output `h'. Shape: `[B, D_out]`
            h_var (Tensor): Element-wise variance of output `h'. Shape: `[B, D_out]`
        """
        
        if a_var == None:
            a_var = torch.zeros_like(a_mean).to(a_mean.device)
            
        h_mean, h_var = forward_aW_diag(a_mean, a_var, self.weight, self.bias, self.w_var, self.b_var)
        
        return h_mean, h_var

class SUQ_Activation_Diag(nn.Module):
    """
    Activation layer with closed-form uncertainty propagation under SUQ, with a diagonal Gaussian posterior.

    Wraps a standard activation function and applies a first-order approximation to propagate input variance through the nonlinearity. See the SUQ paper for theoretical background and assumptions.

    Args:
        afun (Callable): A PyTorch activation function (e.g. `nn.ReLU()`)
    """
    
    def __init__(self, afun):        
        super().__init__()
        self.afun = afun
    
    def forward(self, h_mean, h_var):
        """
        Forward pass with uncertainty propagation through a SUQ activation layer.
        
        Args:
            h_mean (Tensor): Mean of the pre-activations `h`. Shape: `[B, D]`
            h_var (Tensor): Element-wise variance of the pre-activation `h`. Shape: `[B, D]`

        Returns:
            a_mean (Tensor): Mean of the activation `a`. Shape: [B, D]
            a_var (Tensor): Element-wise variance of the activation `a`. Shape: `[B, D]`
        """
        a_mean, a_var = forward_activation_implicit_diag(self.afun, h_mean, h_var)
        return a_mean, a_var

class SUQ_BatchNorm_Diag(nn.Module):
    """
    BatchNorm layer with closed-form uncertainty propagation under SUQ, with a diagonal Gaussian posterior.

    Wraps `nn.BatchNorm1d` and adjusts input variance using batch normalization statistics and scale parameters. See the SUQ paper for theoretical background and assumptions.

    Args:
        BatchNorm (nn.BatchNorm1d): The original batch norm layer
    """
    
    def __init__(self, BatchNorm):
        super().__init__()
        
        self.BatchNorm = BatchNorm
    
    def forward(self, x_mean, x_var):
        """
        Forward pass with uncertainty propagation through a SUQ BatchNorm layer.
        
        Args:
            x_mean (Tensor): Input mean. Shape: [B, D]
            x_var (Tensor): Input element-wise variance. Shape: [B, D]

        Returns:
            out_mean (Tensor): Output mean after batch normalization. Shape: [B, D]
            out_var (Tensor): Output element-wise variance after batch normalization. Shape: [B, D]
        """
        
        with torch.no_grad():
        
            out_mean = self.BatchNorm.forward(x_mean)
            out_var = forward_batch_norm_diag(x_mean, x_var, self.BatchNorm.weight, 1e-5)
            
        return out_mean, out_var


class SUQ_MLP_Diag(SUQ_Base):
    """
    Multilayer perceptron model with closed-form uncertainty propagation under SUQ, with a diagonal Gaussian posterior.

    Wraps a standard MLP, converting its layers into SUQ-compatible components.
    Supports both classification and regression via predictive Gaussian approximation.
    
    Note:
        The input model should correspond to the latent function only:
        - For regression, this is the full model (including final output layer).
        - For classification, exclude the softmax layer and pass only the logit-producing part.

    Args:
        org_model (nn.Module): The original MLP model to convert
        posterior_variance (Tensor): Flattened posterior variance vector
        likelihood (str): Either 'classification' or 'regression'
        scale_init (float, optional): Initial scale factor
        sigma_noise (float, optional): noise level (for regression)
    """
    
    def __init__(self, org_model, posterior_variance, likelihood, scale_init = 1.0, sigma_noise = None):
        super().__init__(likelihood, scale_init)

        self.sigma_noise = sigma_noise
        self.convert_model(org_model, posterior_variance)
    
    def forward_latent(self, data, out_var = None):
        """
        Compute the predictive mean and variance of the latent function before applying the likelihood.

        Traverses the model layer by layer, propagating mean and variance through each SUQ-wrapped layer.

        Args:
            data (Tensor): Input data. Shape: [B, D_in]
            out_var (Tensor or None): Optional input variance. Shape: [B, D_in]

        Returns:
            out_mean (Tensor): Output mean after final layer. Shape: [B, D_out]
            out_var (Tensor): Output element-wise variance after final layer. Shape: [B, D_out]
        """
        
        out_mean = data
        
        if isinstance(self.model, nn.Sequential):
            for layer in self.model:
                out_mean, out_var = layer.forward(out_mean, out_var)
        ##TODO: other type of model            

        out_var = out_var / self.scale_factor
        
        return out_mean, out_var
    
    def forward(self, data):
        """
        Compute the predictive distribution based on the model's likelihood setting.

        For classification, use probit-approximation.
        For regression, returns the latent mean and total predictive variance.

        Args:
            data (Tensor): Input data. Shape: [B, D]

        Returns:
            If classification:
                Tensor: Class probabilities. Shape: [B, num_classes]
            If regression:
                Tuple[Tensor, Tensor]: Output mean and element-wise variance. Shape: [B, D_out]
        """
        
        out_mean, out_var = self.forward_latent(data)

        if self.likelihood == 'classification':
            kappa = 1 / torch.sqrt(1. + np.pi / 8 * out_var)
            return torch.softmax(kappa * out_mean, dim=-1)

        if self.likelihood == 'regression':
            return out_mean, out_var + self.sigma_noise ** 2
    
    def convert_model(self, org_model, posterior_variance):
        """
        Converts a deterministic MLP into a SUQ-compatible model with diagonal posterior.

        Each layer is replaced with its corresponding SUQ module (e.g. linear, activation, batchnorm), using the provided flattened posterior variance vector.

        Args:
            org_model (nn.Module): The original model to convert (latent function only)
            posterior_variance (Tensor): Flattened posterior variance for Bayesian parameters
        """
        
        p_model = copy.deepcopy(org_model)

        loc = 0
        for n, layer in p_model.named_modules():
            if isinstance(layer, nn.Linear):
                
                D_out, D_in = layer.weight.data.shape
                num_param = torch.numel(parameters_to_vector(layer.parameters()))
                num_weight_param = D_out * D_in
                
                covariance_block = posterior_variance[loc : loc + num_param]
                
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
