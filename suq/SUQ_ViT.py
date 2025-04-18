from .diag_suq_transformer import SUQ_ViT_Diag

def streamline_vit(model, posterior, covariance_structure, likelihood, MLP_deterministic, Attn_deterministic, attention_diag_cov, num_det_blocks, scale_init = 1.0):
    if covariance_structure == 'diag':
        return SUQ_ViT_Diag(ViT = model, 
                            posterior_variance = posterior, 
                            MLP_determinstic = MLP_deterministic, 
                            Attn_determinstic = Attn_deterministic, 
                            likelihood = likelihood,
                            attention_diag_cov = attention_diag_cov, 
                            num_det_blocks = num_det_blocks,
                            scale_init = scale_init)
    else:
        raise NotImplementedError(f"Covariance structure '{covariance_structure}' is not implemented.")