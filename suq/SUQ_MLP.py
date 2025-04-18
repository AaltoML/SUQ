from .diag_suq_mlp import SUQ_MLP_Diag

def streamline_mlp(model, posterior, covariance_structure, likelihood, scale_init = 1.0):
    if covariance_structure == 'diag':
        return SUQ_MLP_Diag(org_model = model, 
                            posterior_variance = posterior, 
                            likelihood = likelihood,
                            scale_init = scale_init)
    else:
        raise NotImplementedError(f"Covariance structure '{covariance_structure}' is not implemented.")