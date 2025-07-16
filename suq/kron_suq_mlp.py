import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import copy

from suq.base_suq import SUQ_Base

def forward_aW_kron(a_mean, a_cov, weight, bias, w_kfacs_A, w_kfacs_B, b_cov):
    """
    Compute the mean and covariance of `h = a @ W^T + b` where the posterior has Kronecker product structure B ⊗ A
    
    Note:
        Following the rvec convention, the posterior covariance is Σ = (B ⊗ A +  λ²I)^{-1} where λ²I is the prior precision.
        We do the following approximation for efficient computation: Σ ≈ (B + λI)^{-1} ⊗ (A + λI)^{-1}
        The Kronecker factors passed through should be (B + λI)^{-1} and (A + λI)^{-1}
    
    Args:
        a_mean (Tensor): Mean of the input `a`. Shape: `[B, D_in]`.
        a_cov (Tensor): Covariance of the input `a`. Shape: `[B, D_in, D_in] `
        weight (Tensor): Mean of the weights `W`. Shape: `[D_out, D_in]`
        bias (Tensor): Mean of the bias `b`. Shape: `[D_out, ]`
        w_kfacs_A (Tensor): The A factor of weight's posterior covariance. Shape: `[D_in, D_in]'
        w_kfacs_B (Tensor): The B factor of weight's posterior covariance. Shape: `[D_out, D_out]' 
        b_cov (Tensor): Covariance of the bias `b`. Shape: `[D_out, D_out]`
    
    Returns: 
        h_mean (Tensor): Mean of the pre-activations `h`. Shape: `[B, D_out]`
        h_cov (Tensor): Covariance of the pre-activations `h`. Shape: `[B, D_out, D_out]`
    """
    
    # calculate mean(h)
    h_mean = F.linear(a_mean, weight, bias)
    
    # calculate cov(h)    
    # calculate sum_{i,j} cov(a_i*w_ki, a_j*w_lj)
    aw_cov = torch.einsum('ni, nj, kl, ij-> nkl', a_mean, a_mean, w_kfacs_B, w_kfacs_A) + \
               torch.einsum('ki, lj, nij -> nkl', weight, weight, a_cov) +\
               torch.einsum('nij, kl, ij-> nkl', a_cov, w_kfacs_B, w_kfacs_A)
    
    h_cov = aw_cov + b_cov

    return h_mean, h_cov

def forward_activation_implicit_full(activation_func, h_mean, h_cov):
    
    """
    Approximate the distribution of `a = g(h)` given `h ~ N(h_mean, h_cov)`, where `h_cov` is the covariance of pre-activation `h`.
    Uses a first-order Taylor expansion: `a ~ N(g(h_mean), g'(h_mean)^T @ h_cov @ g'(h_mean))`.

    Args:
        activation_func (Callable): A PyTorch activation function `g(·)` (e.g. `nn.ReLU()`)
        h_mean (Tensor): Mean of the pre-activations `h`. Shape: `[B, D]`
        h_cov (Tensor): Covariance of the pre-activations `h`. Shape: `[B, D, D]`

    Returns:
        a_mean (Tensor): Mean of the activations `a`. Shape: `[B, D]`
        a_cov (Tensor): Covariance of the activations `a`. Shape: `[B, D, D]`
    """

    h_mean_grad = h_mean.detach().clone().requires_grad_()
    
    a_mean = activation_func(h_mean_grad)
    a_mean.retain_grad()
    a_mean.backward(torch.ones_like(a_mean)) #[N, D]
    
    nabla = h_mean_grad.grad #[N, D]
    jacobian = torch.diag_embed(nabla)
    a_cov = torch.einsum('nij, njk, nlk -> nil', jacobian, h_cov, jacobian)
    
    return a_mean.detach(), a_cov

class SUQ_Linear_Kron(nn.Module):
    """
    Linear layer with uncertainty propagation under SUQ, with a Kron Gaussian posterior.
    
    Wraps a standard `nn.Linear` layer and applies closed-form mean and covariance propagation. See the SUQ paper for theoretical background and assumptions.

    Args:
        org_linear (nn.Linear): The original linear layer to wrap      
        w_kfacs_A (Tensor): The A factor of weight's posterior covariance. Shape: `[D_in, D_in]'
        w_kfacs_B (Tensor): The B factor of weight's posterior covariance. Shape: `[D_out, D_out]' 
        b_cov (Tensor): Covariance of the bias `b`. Shape: `[D_out, D_out]`
    """
    def __init__(self, org_linear, w_kfacs_A, w_kfacs_B, b_cov):
        super().__init__()
        
        self.weight = org_linear.weight.data
        self.bias = org_linear.bias.data
        self.w_kfacs_A = w_kfacs_A
        self.w_kfacs_B = w_kfacs_B
        self.b_cov = b_cov
    
    def forward(self, a_mean, a_cov): 
        """
        Forward pass with uncertainty propagation through a SUQ linear layer.
        
        Args:
            a_mean (Tensor): Input mean. Shape: `[B, D_in]`
            a_cov (Tensor): Input covariance. Shape: `[B, D_in, D_in]`

        Returns:
            h_mean (Tensor): Output mean. Shape: `[B, D_out]`
            h_cov (Tensor): Output covariance. Shape: `[B, D_out, D_out]`
        """
        
        if a_cov== None:
            a_cov = torch.zeros((a_mean.shape[0], a_mean.shape[1], a_mean.shape[1]), device = a_mean.device)

        h_mean, h_cov = forward_aW_kron(a_mean, a_cov, self.weight, self.bias, self.w_kfacs_A, self.w_kfacs_B, self.b_cov)
        
        return h_mean, h_cov

class SUQ_Activation_Full(nn.Module):
    """
    Activation layer with closed-form uncertainty propagation under SUQ, with a full Gaussian posterior.

    Wraps a standard activation function and applies a first-order approximation to propagate input variance through the nonlinearity. See the SUQ paper for theoretical background and assumptions.

    Args:
        afun (Callable): A PyTorch activation function (e.g. `nn.ReLU()`)
    """
    
    def __init__(self, afun):        
        super().__init__()
        self.afun = afun
    
    def forward(self, h_mean, h_cov):
        """
        Forward pass with uncertainty propagation through a SUQ activation layer.
        
        Args:
            h_mean (Tensor): Mean of the pre-activations `h`. Shape: `[B, D]`
            h_var (Tensor): Element-wise variance of the pre-activation `h`. Shape: `[B, D]`

        Returns:
            a_mean (Tensor): Mean of the activation `a`. Shape: [B, D]
            a_var (Tensor): Element-wise variance of the activation `a`. Shape: `[B, D]`
        """
        a_mean, a_cov = forward_activation_implicit_full(self.afun, h_mean, h_cov)
        return a_mean, a_cov

class SUQ_MLP_Kron(SUQ_Base):
    """
    Multilayer perceptron model with closed-form uncertainty propagation under SUQ, with a Kronecker Gaussian posterior.

    Wraps a standard MLP, converting its layers into SUQ-compatible components.
    Supports both classification and regression via predictive Gaussian approximation.
    
    Note:
        Currently the wrapper only support the posterior which has the structure as Laplace redux library.
        
    Args:
        org_model (nn.Module): The original MLP model to convert
        hessian_eigenvector (list): Eigenvectors of Kronecker factors, corresponds to `la.H.eigenvector` 
        hessian_eigenvalue (list): Eigenvalues of Kronecker factors, corresponds to `la.H.eigenvalue` 
        prior_precision (float): prior precision
        likelihood (str): Either 'classification' or 'regression'
        scale_init (float, optional): Initial scale factor
        sigma_noise (float, optional): noise level (for regression)
    """
    
    def __init__(self, org_model, hessian_eigenvector, hessian_eigenvalue, likelihood, prior_precision, scale_init = 1.0, sigma_noise = None):
        super().__init__(likelihood, scale_init)

        self.sigma_noise = sigma_noise
        self.convert_model(org_model, hessian_eigenvector, hessian_eigenvalue, prior_precision)
    
    def forward_latent(self, data, out_cov = None):
        """
        Compute the predictive mean and covariance of the latent function before applying the likelihood.

        Traverses the model layer by layer, propagating mean and covariance through each SUQ-wrapped layer.

        Args:
            data (Tensor): Input data. Shape: [B, D_in]
            out_cov (Tensor or None): Optional input covariance. Shape: [B, D_in]

        Returns:
            out_mean (Tensor): Output mean after final layer. Shape: [B, D_out]
            out_cov (Tensor): Output covariance after final layer. Shape: [B, D_out]
        """
        
        out_mean = data
        
        if isinstance(self.model, nn.Sequential):
            for layer in self.model:
                out_mean, out_var = layer.forward(out_mean, out_cov)
        ##TODO: other type of model            

        out_cov = out_cov / self.scale_factor
        
        return out_mean, out_cov
    
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
        
        out_mean, out_cov = self.forward_latent(data)

        if self.likelihood == 'classification':
            kappa = 1 / torch.sqrt(1. + np.pi / 8 * out_cov.diagonal(dim1=1, dim2=2))
            return torch.softmax(kappa * out_mean, dim=-1)

        if self.likelihood == 'regression':
            return out_mean, torch.diagonal(out_cov, dim1=1, dim2=2) + self.sigma_noise ** 2
    
    def convert_model(self, org_model, hessian_eigenvector, hessian_eigenvalue, prior_precision):
        """
        Converts a deterministic MLP into a SUQ-compatible model with Kronecker posterior.

        Each layer is replaced with its corresponding SUQ module (e.g. linear, activation), using the provided covariance.

        Args:
            org_model (nn.Module): The original model to convert (latent function only)
            hessian_eigenvector (list): Eigenvectors of Kronecker factors, corresponds to `la.H.eigenvector` 
            hessian_eigenvalue (list): Eigenvalues of Kronecker factors, corresponds to `la.H.eigenvalue` 
            prior_precision (float): prior precision
        """
        
        p_model = copy.deepcopy(org_model)
        prior_precision_sqrt = np.sqrt(prior_precision)

        loc = 0
        for n, layer in p_model.named_modules():
            if isinstance(layer, nn.Linear):
                
                Q_B, Q_A = hessian_eigenvector[loc]
                l_B = copy.deepcopy(hessian_eigenvalue[loc][0])
                l_A = copy.deepcopy(hessian_eigenvalue[loc][1])

                l_A += prior_precision_sqrt
                l_B += prior_precision_sqrt
                
                w_precision_kfacs_A = Q_A @ torch.diag_embed(l_A) @ Q_A.T
                w_precision_kfacs_B = Q_B @ torch.diag_embed(l_B) @ Q_B.T
                b_H = hessian_eigenvector[loc+1][0] @ torch.diag_embed(hessian_eigenvalue[loc+1][0]) @ hessian_eigenvector[loc+1][0].T
                b_cov = torch.linalg.inv(b_H + prior_precision * torch.eye(l_B.shape[0], device = l_A.device))
                    
                w_cov_kfacs_A = torch.linalg.inv(w_precision_kfacs_A)
                w_cov_kfacs_B = torch.linalg.inv(w_precision_kfacs_B)

                new_layer = SUQ_Linear_Kron(layer, w_cov_kfacs_A, w_cov_kfacs_B, b_cov)
                
                loc += 2

                setattr(p_model, n, new_layer)   



            if type(layer).__name__ in torch.nn.modules.activation.__all__:
                new_layer = SUQ_Activation_Full(layer)
                setattr(p_model, n, new_layer)

        self.model = p_model
