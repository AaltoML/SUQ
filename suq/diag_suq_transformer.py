import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from suq.diag_suq_mlp import forward_aW_diag
from suq.base_suq import SUQ_Base

def forward_linear_diag_Bayesian_weight(e_mean, e_var, w_mean, w_var, bias = None):
    """
    Compute the mean and element-wise variance of `h = e @ W^T + b` when `e ~ N(e_mean, e_var)` and `W ~ N(w_mean, w_var)`
    
    Note:
        - We only make the weight Bayesian, bias is treated determinstically
        - We always assume the input to next layer has diagonal covariance, so we only compute the variance over `h` here.
    
    Args:
        e_mean (Tensor): Mean of the input embeddings `e`. Shape: `[B, T, D_in]`
        e_var (Tensor): Element-wise variance of the input embeddings `e`. Shape: `[B, T, D_in]`
        w_mean (Tensor): Mean of the weights `W`. Shape: `[D_out, D_in]`
        w_var (Tensor): Element-wise variance of the weights `W`. Shape: `[D_out, D_in]`
        bias (Tensor, optional): Bias term `b`. Shape: `[D_out,]`

    Returns:
        h_mean (Tensor): Mean of the output `h`. Shape: `[B, T, D_out]`
        h_var (Tensor): Element-wise variance of the output `h`. Shape: `[B, T, D_out]`
    """

    # calculate mean(h)
    h_mean = F.linear(e_mean, w_mean, bias)
    
    # calculate var(h)
    weight_mean2_var_sum = w_mean ** 2 + w_var # [D_out, D_in]
    h_var = e_mean **2 @ w_var.T + e_var @ weight_mean2_var_sum.T

    return h_mean, h_var

def forward_linear_diag_determinstic_weight(e_mean, e_var, weight, bias = None):
    """
    Compute the mean and element-wise variance of `h = e @ W^T + b` when `e ~ N(e_mean, e_var)`, `W` and `b` are both determinstic
    
    Note:
        - We always assume the input to next layer has diagonal covariance, so we only compute the variance over `h` here.
    
    Args:
        e_mean (Tensor): Mean of the input embeddings `e`. Shape: `[B, T, D_in]`.
        e_var (Tensor): Element-wise variance of the input embeddings `e`. Shape: `[B, T, D_in]`.
        weight (Tensor): Weights `W`. Shape: `[D_out, D_in]`.
        bias (Tensor, optional): Bias term `b`. Shape: `[D_out,]`.
        
    Returns:
        h_mean (Tensor): Mean of the output `h`. Shape: `[B, T, D_out]`
        h_var (Tensor): Element-wise variance of the output `h`. Shape: `[B, T, D_out]`
    """
    
    h_mean = F.linear(e_mean, weight, bias)
    h_var = F.linear(e_var, weight ** 2, None)
    
    return h_mean, h_var

@torch.enable_grad()
def forward_activation_diag(activation_func, h_mean, h_var):
    """
    Approximate the distribution of `a = g(h)` given `h ~ N(h_mean, h_var)`, where `h_var` 
    is the element-wise variance of pre-activation `h`.
    Uses a first-order Taylor expansion: `a ~ N(g(h_mean), g'(h_mean)^T @ h_var @ g'(h_mean))`.
    
    Args:
        activation_func (Callable): A PyTorch activation function `g(·)` (e.g. `nn.ReLU()`)
        h_mean (Tensor): Mean of the pre-activations `h`. Shape: `[B, T, D]`
        h_var (Tensor): Element-wise variance of the pre-activations `h`. Shape: `[B, T, D]`

    Returns:
        a_mean (Tensor): Mean of the activations `a`. Shape: `[B, T, D]`
        a_var (Tensor): Element-wise variance of the activations `a`. Shape: `[B, T, D]`
    """

    h_mean_grad = h_mean.detach().clone().requires_grad_()
    
    a_mean = activation_func(h_mean_grad)
    a_mean.retain_grad()
    a_mean.backward(torch.ones_like(a_mean)) #[B, T, D]
    
    nabla = h_mean_grad.grad #[B, T, D]
    a_var = nabla ** 2 * h_var
    
    return a_mean.detach(), a_var
    
def forward_layer_norm_diag(e_mean, e_var, ln_weight, ln_eps):
    """
    Compute the output variance when a distribution `e ~ N(e_mean, e_var)`
    is passed through a LayerhNorm layer.
    
    Args:
        e_mean (Tensor): Mean of the input distribution. Shape: `[B, T, D]`
        e_var (Tensor): Element-wise variance of the input distribution. Shape: `[B, T, D]`
        ln_weight (Tensor): LayerNorm scale factor (gamma). Shape: `[D,]`
        ln_eps (float): Small constant added for numerical stability.

    Returns:
        output_var (Tensor): Element-wise variance after LayerNorm. Shape: `[B, T, D]`
    """

    # calculate the var
    input_mean_var = e_mean.var(dim=-1, keepdim=True, unbiased=False) # [B, T, 1]
    scale_factor = (1 / (input_mean_var + ln_eps)) * ln_weight **2 # [B, T, D]
    output_var = scale_factor * e_var # [B, T, D]
    
    return output_var

def forward_value_cov_Bayesian_W(W_v, W_v_var, e_mean, e_var, n_h, D_v, diag_cov = False):
    """
    Given value weight W_v ~ N(W_v, W_v_var) and input E ~ N(e_mean, e_var)
    Compute the covariance of output `v = W_v @ E `
    
    Args:
        n_h (int): Number of attention heads.
        D_v (int): Dimension per head. Must satisfy `n_h * D_v = D`
        W_v (Tensor): Mean of the value weight `W_v`. Shape: `[D, D]`
        W_v_var (Tensor): Element-wise variance of the value weight `W_v`. Shape: `[D, D]`
        e_mean (Tensor): Mean of the input embeddings `e`. Shape: `[B, T, D]`
        e_var (Tensor): Element-wise variance of the input embeddings `e`. Shape: `[B, T, D]`
        diag_cov (bool): If `True`, only compute and return diagonal of the output covariance.

    Returns:
        v_var (Tensor): Returned if `diag_cov=True`. Element-wise variance of the output `v`.  Shape: `[B, T, n_h, D_v]`
        v_cov (Tensor): Returned if `diag_cov=False`. Full covariance matrices of the output `v`.  Shape: `[B, T, n_h, D_v, D_v]`
    """

    B, T, D = e_var.size()

    if not diag_cov:
        ## compute general covariance 
        W_v_reshaped = W_v.reshape(1, 1, n_h, D_v, D) 
            # [D, D] -> [1, 1, n_h, D_v, D]
        input_var_reshaped = e_var.reshape(B, T, 1, 1, D)
            # [B, T, D] -> [B, T, 1, 1, D]
        v_cov = (W_v_reshaped * input_var_reshaped).transpose(3, 4)
            # [1, 1, n_h, D_v, D] * [B, T, 1, 1, D] -> [B, T, n_h, D_v, D] -> [B, T, n_h, D, D_v]
        v_cov = torch.matmul(W_v_reshaped, v_cov)
            #  [1, 1, n_h, D_v, D] @ [B, T, n_h, D, D_v]  -> [B, T, n_h, D_v, D_v]
        
        ## add missing part for variance
        W_v_var_reshaped = W_v_var.reshape(1, 1, n_h, D_v, D) 
            #[D, D] -> [1, 1, n_h, D_v, D]
        input_var_plus_mean_square = input_var_reshaped + e_mean.reshape(B, T, 1, 1, D)**2 #[B, T, 1, 1, D]
        extra_var_term = torch.sum(input_var_plus_mean_square * W_v_var_reshaped, dim=[4]) # [B, T, n_h, D_v, D] -> [B, T, n_h, D_v]
        v_cov = v_cov + torch.diag_embed(extra_var_term) 

        return v_cov

    else:
        weight_mean2_var_sum = W_v **2 + W_v_var # [D, D]
        v_var = e_mean **2 @ W_v_var.T + e_var @ weight_mean2_var_sum.T # [B, T, D]
    
        return v_var.reshape(B, T, n_h, D_v)

def forward_value_cov_determinstic_W(W_v, e_var, n_h, D_v, diag_cov = False):
    """
    Given determinstic value weight W_v and input E ~ N(e_mean, e_var)
    Compute the covariance of output `v = W_v @ E`

    Args:
        n_h (int): Number of attention heads.
        D_v (int): Dimension per head. Must satisfy `n_h * D_v = D`
        W_v (Tensor): Value weight `W_v`. Shape: `[D, D]`
        e_var (Tensor): Element-wise variance of the input embeddings `e`. Shape: `[B, T, D]`
        diag_cov (bool): If `True`, only compute and return diagonal of the output covariance.

    Returns:
        v_var (Tensor): Returned if `diag_cov=True`. Element-wise variance of the output `v`.  Shape: `[B, T, n_h, D_v]`
        v_cov (Tensor): Returned if `diag_cov=False`. Full covariance matrices of the output `v`.  Shape: `[B, T, n_h, D_v, D_v]`

    """

    B, T, D = e_var.size()

    if not diag_cov:
        W_v_reshaped = W_v.reshape(1, 1, n_h, D_v, D) 
            #[n_h, D_v, D] -> [1, 1, n_h, D_v, D]
        input_var_reshaped = e_var.reshape(B, T, 1, 1, D)
            # [B, T, D] -> [B, T, 1, 1, D]
        v_cov = (W_v_reshaped * input_var_reshaped).transpose(3, 4)
            # [1, 1, n_h, D_v, D] * [B, T, 1, 1, D] -> [B, T, n_h, D_v, D] -> [B, T, n_h, D, D_v]
        v_cov = torch.matmul(W_v_reshaped, v_cov)
            #  [1, 1, n_h, D_v, D] @ [B, T, n_h, D, D_v]  -> [B, T, n_h, D_v, D_v]

        return v_cov
    
    else:
        v_var = e_var @ (W_v ** 2).T

        return v_var.reshape(B, T, n_h, D_v)
    
def forward_QKV_cov(attention_score, v_cov, diag_cov = False):
    """
    Given attention score (QK^T) and `V ~ N(v_mean, v_cov)`
    Compute the covariance of output `E = (QK^T) V`
    
    Args:
        attention_score (Tensor): Attention weights `A = QK^T`. Shape: `[B, n_h, T, T]`
        v_cov (Tensor): Covariance of the value `V`.  Shape: `[B, T, n_h, D_v, D_v]` if `diag_cov=False`. `[B, T, n_h, D_v]` if `diag_cov=True`
        diag_cov (bool): If `True`, value `V` will have diagonal covariance

    Returns:
        QKV_var (Tensor): Returned if `diag_cov=True`. Element-wise variance of the output `E`. Shape: `[B, T, n_h, D_v]`
        QKV_cov (Tensor): Returned if `diag_cov=False`. Full covariance matrices of the output `E`. Shape: `[B, n_h, T, D_v, D_v]`
    """
    if diag_cov:
        B, T, n_h, D_v = v_cov.size()
        QKV_cov = attention_score **2 @ v_cov.transpose(1, 2) # [B, n_h, T, D_v]
            # v_cov [B, T, n_h, D_v] -> [B, n_h, T, D_v]
            # [B, n_h, T, T] @ [B, n_h, T, D_v]  -> [B, n_h, T, D_v]
    else:
        
        B, T, n_h, D_v, _ = v_cov.size()
        
        QKV_cov = attention_score **2 @ v_cov.permute(0, 2, 1, 3, 4).reshape(B, n_h, T, D_v * D_v) # [B, n_h, T, D_v * D_v]
        # v_cov [B, T, n_h, D_v, D_v] -> [B, n_h, T, D_v * D_v]
        # [B, n_h, T, T] @ [B, n_h, T, D_v * D_v]  -> [B, n_h, T, D_v * D_v]
        QKV_cov = QKV_cov.reshape(B, n_h, T, D_v, D_v)
        
    return QKV_cov

def forward_fuse_multi_head_cov(QKV_cov, project_W, diag_cov = False):
    """
    Given concatanated multi-head embedding `E ~ N(e_mean, e_cov)` and the determinstic projection weight matrix `W`
    Compute variance of each output dimenison 
    
    Args:
        QKV_cov (Tensor): Covariance of the concatenated multi-head output `E`.  Shape: `[B, T, n_h, D_v, D_v]` if `diag_cov=False`. `[B, T, n_h, D_v]` if `diag_cov=True`
        project_W (Tensor): Projection weight matrix `W`. Shape: `[D_out, D_in]`, where `D_in = n_h * D_v`
        diag_cov (bool): If `True`, `QKV_cov` will have diagonal covariance
        
    Returns: 
        output_var (Tensor): Element-wise variance of the projected output. Shape: `[B, T, D_out]`
    """
    if diag_cov:
        B, n_h, T, D_v = QKV_cov.size()
        output_var = QKV_cov.permute(0, 2, 1, 3).reshape(B, T, n_h * D_v) @ project_W ** 2
            # QKV_cov [B, n_h, T, D_v] -> [B, T, n_h, D_v] -> [B, T, n_h * D_v]

        return output_var
        
    else:
        B, n_h, T, D_v, _ = QKV_cov.size()
        D, _ = project_W.shape
        
        project_W_reshaped_1 = project_W.T.reshape(n_h, D_v, D).permute(0, 2, 1).reshape(n_h * D, D_v, 1)
            # [n_h, D_v, D] -> [n_h, D, D_v] -> [n_h * D, D_v, 1]
        project_W_reshaped_2 = project_W.T.reshape(n_h, D_v, D).permute(0, 2, 1).reshape(n_h * D, 1, D_v)
            # [n_h, D_v, D] -> [n_h, D, D_v] -> [n_h * D, 1, D_v]

        project_W_outer = torch.bmm(project_W_reshaped_1, project_W_reshaped_2).reshape(n_h, D, D_v, D_v).permute(1, 0, 2, 3) # [D, n_h, D_v, D_v]
        # [n_h * D, D_v, D_v] @ [n_h * D, 1, D_v] -> [n_h * D, D_v, D_v] -> [D, n_h, D_v, D_v]
            
        output_var_einsum = torch.einsum('dhij,bthij->dbt', project_W_outer, QKV_cov.permute(0, 2, 1, 3, 4))
        
        return output_var_einsum.permute(1, 2, 0)

class SUQ_LayerNorm_Diag(nn.Module):
    """
    LayerNorm module with uncertainty propagation under SUQ.

    Wraps `nn.LayerNorm` and propagates input variance analytically using running statistics. See the SUQ paper for theoretical background and assumptions.

    Args:
        LayerNorm (nn.LayerNorm): The original layer norm module to wrap
    """

    def __init__(self, LayerNorm):
        super().__init__()
        
        self.LayerNorm = LayerNorm
    
    def forward(self, x_mean, x_var):
        """
        Forward pass with uncertainty propagation through a SUQ LayerNorm layer.
        
        Args:
            x_mean (Tensor): Input mean. Shape: [B, T, D]
            x_var (Tensor): Input element-wise variance. Shape: [B, T, D]

        Returns:
            out_mean (Tensor): Output mean after layer normalization. Shape: [B, T, D]
            out_var (Tensor): Output element-wise variance after layer normalization. Shape: [B, T, D]
        """
        
        with torch.no_grad():
        
            out_mean = self.LayerNorm.forward(x_mean)
            out_var = forward_layer_norm_diag(x_mean, x_var, self.LayerNorm.weight, 1e-5)
            
        return out_mean, out_var


class SUQ_Classifier_Diag(nn.Module):
    """
    Classifier head with uncertainty propagation under SUQ, with a diagonal Gaussian posterior.

    Wraps a standard linear classifier and applies closed-form mean and variance propagation.
    See the SUQ paper for theoretical background and assumptions.

    Args:
        classifier (nn.Linear): The final classification head
        w_var (Tensor): Element-wise variance of weight. Shape: `[D_out, D_in]`
        b_var (Tensor): Element-wise variance of bias. Shape: `[D_out]`
    """

    def __init__(self, classifier, w_var, b_var):
        super().__init__()
        
        self.weight = classifier.weight
        self.bias = classifier.bias
        self.w_var = w_var.reshape(self.weight.shape)
        self.b_var = b_var.reshape(self.bias.shape)
    
    def forward(self, x_mean, x_var):
        """
        Forward pass with uncertainty propagation through a SUQ linear layer.

        Args.
            x_mean (Tensor): Input mean. Shape: `[B, D_in]`
            x_var (Tensor): Input element-wise variance. Shape: `[B, D_in]`

        Returns:
            h_mean (Tensor): Output mean. Shape: `[B, D_out]`
            h_var (Tensor): Output element-wise variance. Shape: `[B, D_out]`
        """
        with torch.no_grad():
            h_mean, h_var = forward_aW_diag(x_mean, x_var, self.weight.data, self.bias.data, self.w_var, self.b_var)
        return h_mean, h_var

class SUQ_TransformerMLP_Diag(nn.Module):
    """
    MLP submodule of a transformer block with uncertainty propagation under SUQ.

    Supports both deterministic and Bayesian forward modes with closed-form variance propagation.
    Used internally in `SUQ_Transformer_Block_Diag`.

    Args:
        MLP (nn.Module): Original MLP submodule
        determinstic (bool): Whether to treat the MLP weights as deterministic
        w_fc_var (Tensor, optional): Variance of the first linear layer in MLP(if Bayesian)
        w_proj_var (Tensor, optional): Variance of the second linear layer in MLP (if Bayesian)
    """

    def __init__(self, MLP, determinstic = True, w_fc_var = None, w_proj_var = None):
        super().__init__()

        self.MLP = MLP
        self.determinstic = determinstic
        if not determinstic:
            self.w_fc_var = w_fc_var.reshape(self.MLP.c_fc.weight.shape)
            self.w_proj_var = w_proj_var.reshape(self.MLP.c_proj.weight.shape)

    def forward(self, x_mean, x_var):
        """
        Forward pass with uncertainty propagation through a SUQ Transformer MLP layer.
        
        Args:
            x_mean (Tensor): Input mean. Shape [B, T, D]
            x_var (Tensor): Input element-wise variance. Shape [B, T, D]

        Returns:
            h_mean (Tensor): Output mean. Shape [B, T, D]
            h_var (Tensor): Output element-wise variance. Shape [B, T, D]
        """
        
        # first fc layer
        with torch.no_grad():
            if self.determinstic:
                h_mean, h_var = forward_linear_diag_determinstic_weight(x_mean, x_var, self.MLP.c_fc.weight.data, self.MLP.c_fc.bias.data)
            else:
                h_mean, h_var = forward_linear_diag_Bayesian_weight(x_mean, x_var, self.MLP.c_fc.weight.data, self.w_fc_var, self.MLP.c_fc.bias.data)
        # activation function
        h_mean, h_var = forward_activation_diag(self.MLP.gelu, h_mean, h_var)
        # second fc layer
        with torch.no_grad():
            if self.determinstic:
                h_mean, h_var = forward_linear_diag_determinstic_weight(h_mean, h_var, self.MLP.c_proj.weight.data, self.MLP.c_proj.bias.data)
            else:
                h_mean, h_var = forward_linear_diag_Bayesian_weight(h_mean, h_var, self.MLP.c_proj.weight.data, self.w_proj_var, self.MLP.c_proj.bias.data)

        return h_mean, h_var

class SUQ_Attention_Diag(nn.Module):
    """
    Self-attention module with uncertainty propagation under SUQ.

    Supports deterministic and Bayesian value projections, with optional diagonal covariance assumptions. For details see SUQ paper section A.6
    Used internally in `SUQ_Transformer_Block_Diag`.

    Args:
        Attention (nn.Module): The original attention module
        determinstic (bool): Whether to treat value projections as deterministic
        diag_cov (bool): If True, only compute the diagoanl covariance for value
        W_v_var (Tensor, optional): Posterior variance for value matrix (if Bayesian)
    """
    
    def __init__(self, Attention, determinstic = True, diag_cov = False, W_v_var = None):
        super().__init__()
        
        self.Attention = Attention
        self.determinstic = determinstic
        self.diag_cov = diag_cov
        
        if not self.determinstic:
            self.W_v_var = W_v_var # [D * D]

    def forward(self, x_mean, x_var):
        """
        Forward pass with uncertainty propagation through a SUQ Attention layer.
        
        Args:
            x_mean (Tensor): Input mean. Shape [B, T, D]
            x_var (Tensor): Input element-wise variance. Shape [B, T, D]

        Returns:
            output_mean (Tensor): Output mean. Shape [B, T, D]
            output_var (Tensor): Output element-wise variance. Shape [B, T, D]
        """

        with torch.no_grad():
        
            output_mean, attention_score = self.Attention.forward(x_mean, True)
            
            n_h = self.Attention.n_head
            B, T, D = x_mean.size()
            D_v = D // n_h
            
            W_v = self.Attention.c_attn_v.weight.data
            project_W = self.Attention.c_proj.weight.data

            if self.determinstic:
                v_cov = forward_value_cov_determinstic_W(W_v, x_var, n_h, D_v)
            else:
                v_cov = forward_value_cov_Bayesian_W(W_v, self.W_v_var.reshape(D, D), x_mean, x_var, n_h, D_v, self.diag_cov)

            QKV_cov = forward_QKV_cov(attention_score, v_cov, self.diag_cov)
            output_var = forward_fuse_multi_head_cov(QKV_cov, project_W, self.diag_cov)

            return output_mean, output_var
        
class SUQ_Transformer_Block_Diag(nn.Module):
    """
    Single transformer block with uncertainty propagation under SUQ.

    Wraps LayerNorm, attention, and MLP submodules with uncertainty-aware versions.
    Used in `SUQ_ViT_Diag` to form a full transformer stack.

    Args:
        MLP (nn.Module): Original MLP submodule
        Attention (nn.Module): Original attention submodule
        LN_1 (nn.LayerNorm): Pre-attention layer norm
        LN_2 (nn.LayerNorm): Pre-MLP layer norm
        MLP_determinstic (bool): Whether to treat MLP as deterministic
        Attn_determinstic (bool): Whether to treat attention as deterministic
        diag_cov (bool): If True, only compute the diagoanl covariance for value
        w_fc_var (Tensor or None): Posterior variance of MLP input projection (if Bayesian)
        w_proj_var (Tensor or None): Posterior variance of MLP output projection (if Bayesian)
        W_v_var (Tensor or None): Posterior variance of value matrix (if Bayesian)
    """

    
    def __init__(self, MLP, Attention, LN_1, LN_2, MLP_determinstic, Attn_determinstic, diag_cov = False, w_fc_var = None, w_proj_var = None, W_v_var = None):
        super().__init__()
        
        self.ln_1 = SUQ_LayerNorm_Diag(LN_1)
        self.ln_2 = SUQ_LayerNorm_Diag(LN_2)
        self.attn = SUQ_Attention_Diag(Attention, Attn_determinstic, diag_cov, W_v_var)
        self.mlp = SUQ_TransformerMLP_Diag(MLP, MLP_determinstic, w_fc_var, w_proj_var)

    def forward(self, x_mean, x_var):
        """
        Forward pass with uncertainty propagation through a SUQ Transformer block.    

        Args:
            x_mean (Tensor): Input mean. Shape [B, T, D]
            x_var (Tensor): Input element-wise variance. Shape [B, T, D]

        Returns:
            h_mean (Tensor): Output mean. Shape [B, T, D]
            h_var (Tensor): Output element-wise variance. Shape [B, T, D]
        """
        
        h_mean, h_var = self.ln_1(x_mean, x_var)
        h_mean, h_var = self.attn(h_mean, h_var)
        h_mean = h_mean + x_mean
        h_var = h_var + x_var
        
        old_h_mean, old_h_var = h_mean, h_var
        
        h_mean, h_var = self.ln_2(h_mean, h_var)
        h_mean, h_var = self.mlp(h_mean, h_var)
        h_mean = h_mean + old_h_mean
        h_var = h_var + old_h_var
        
        return h_mean, h_var


class SUQ_ViT_Diag(SUQ_Base):
    """
    Vision Transformer model with uncertainty propagation under SUQ, with a diagonal Gaussian posterior.

    Wraps a ViT architecture into a structured uncertainty-aware model by replacing parts
    of the network with SUQ-compatible blocks. Allows selective Bayesian treatment of MLP
    and attention modules within each transformer block.

    Currently supports classification only. See the SUQ paper for theoretical background and assumptions.

    Args:
        ViT (nn.Module): A Vision Transformer model structured like `examples/vit_model.py`
        posterior_variance (Tensor): Flattened posterior variance vector
        MLP_determinstic (bool): Whether MLP submodules are treated as deterministic
        Attn_determinstic (bool): Whether attention submodules are treated as deterministic
        scale_init (float, optional): Initial value for the scale factor
        attention_diag_cov (bool): If True, only compute the diagoanl covariance for value
        likelihood (str): Currently only support 'Classification'
        num_det_blocks (int): Number of transformer blocks to leave deterministic (from the bottom up)
    """

    def __init__(self, ViT, posterior_variance, MLP_determinstic, Attn_determinstic, scale_init = 1.0, attention_diag_cov = False, likelihood = 'clasification', num_det_blocks = 10):
        super().__init__(likelihood, scale_init)
        
        if likelihood not in ['classification']:
            raise ValueError(f"{likelihood} not supported for ViT")
        
    
        self.transformer = nn.ModuleDict(dict(
            pte = ViT.transformer.pte,
            h = nn.ModuleList(),
            ln_f = SUQ_LayerNorm_Diag(ViT.transformer.ln_f)
        ))
        
        self.scale_factor = nn.Parameter(torch.Tensor([scale_init]))
        
        num_param_c_fc = ViT.transformer.h[0].mlp.c_fc.weight.numel()
        num_param_c_proj = ViT.transformer.h[0].mlp.c_proj.weight.numel()
        num_param_value_matrix = ViT.transformer.h[0].attn.c_proj.weight.numel()
        
        index = 0
        for block_index in range(len(ViT.transformer.h)):
            
            if block_index < num_det_blocks:
                self.transformer.h.append(ViT.transformer.h[block_index])
            else:
                if not MLP_determinstic:
                    w_fc_var = posterior_variance[index: index + num_param_c_fc]
                    index += num_param_c_fc
                    w_proj_var = posterior_variance[index: index + num_param_c_proj]
                    index += num_param_c_proj
                    self.transformer.h.append(
                        SUQ_Transformer_Block_Diag(ViT.transformer.h[block_index].mlp, 
                                                    ViT.transformer.h[block_index].attn, 
                                                    ViT.transformer.h[block_index].ln_1, 
                                                    ViT.transformer.h[block_index].ln_2, 
                                                    MLP_determinstic,
                                                    Attn_determinstic,
                                                    attention_diag_cov,
                                                    w_fc_var, 
                                                    w_proj_var,
                                                    None))
                
                if not Attn_determinstic:
                    w_v_var = posterior_variance[index : index + num_param_value_matrix]
                    index += num_param_value_matrix
                    self.transformer.h.append(
                        SUQ_Transformer_Block_Diag(ViT.transformer.h[block_index].mlp, 
                                                    ViT.transformer.h[block_index].attn, 
                                                    ViT.transformer.h[block_index].ln_1, 
                                                    ViT.transformer.h[block_index].ln_2, 
                                                    MLP_determinstic,
                                                    Attn_determinstic,
                                                    attention_diag_cov,
                                                    None, 
                                                    None,
                                                    w_v_var))

        num_param_classifier_weight = ViT.classifier.weight.numel()
        self.classifier = SUQ_Classifier_Diag(ViT.classifier, posterior_variance[index: index + num_param_classifier_weight], posterior_variance[index + num_param_classifier_weight:])

    def forward_latent(self, pixel_values, interpolate_pos_encoding = None):
        
        """
        Compute the predictive mean and variance of the ViT's latent output before applying the final likelihood layer.

        Traverses the full transformer stack with uncertainty propagation.

        Args:
            pixel_values (Tensor): Input image tensor, shape [B, C, H, W]
            interpolate_pos_encoding (optional): Optional positional embedding interpolation

        Returns:
            x_mean (Tensor): Predicted latent mean at the [CLS] token, shape [B, D]
            x_var (Tensor): Predicted latent variance at the [CLS] token, shape [B, D]
        """

        device = pixel_values.device

        x_mean = self.transformer.pte(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )
        
        # pass through model
        x_var = torch.zeros_like(x_mean, device = device)
        
        for i, block in enumerate(self.transformer.h):
            
            if isinstance(block, SUQ_Transformer_Block_Diag):
            
                x_mean, x_var = block(x_mean, x_var)
            else:
                x_mean = block(x_mean)

        x_mean, x_var = self.transformer.ln_f(x_mean, x_var)
        
        x_mean, x_var = self.classifier(x_mean[:, 0, :], x_var[:, 0, :])
        x_var = x_var / self.scale_factor.to(device)
        
        return x_mean, x_var

    def forward(self, pixel_values, interpolate_pos_encoding = None):
        """
        Compute predictive class probabilities using a probit approximation.

        Performs a full forward pass through the ViT with uncertainty propagation, and
        produces softmax-normalized class probabilities for classification.

        Args:
            pixel_values (Tensor): Input image tensor, shape [B, C, H, W]
            interpolate_pos_encoding (optional): Optional positional embedding interpolation

        Returns:
            Tensor: Predicted class probabilities, shape [B, num_classes]
        """

        x_mean, x_var = self.forward_latent(pixel_values, interpolate_pos_encoding)
        kappa = 1 / torch.sqrt(1. + np.pi / 8 * x_var)
        
        return torch.softmax(kappa * x_mean, dim=-1)

