import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.distributions import Categorical
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader

from suq.utils.utils import torch_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SUQ_Base(nn.Module):
    """
    Base class for SUQ models.

    Provides core functionality for:
    - Managing likelihood type (regression or classification)
    - Probit-based approximation for classification
    - NLPD-based fitting of the scale factor

    Inputs:
        likelihood (str): Either 'classification' or 'regression'
        scale_init (float): Initial value for the scale factor parameter
    """
    
    def __init__(self, likelihood, scale_init):
        super().__init__()
        
        if likelihood not in ['classification', 'regression']:
            raise ValueError(f"Invalid likelihood type {likelihood}")
        
        self.likelihood = likelihood
        self.scale_factor = nn.Parameter(torch.Tensor([scale_init]).to(device))
    
    def probit_approximation(self, out_mean, out_var):
        """
        Applies a probit approximation to compute class probabilities from the latent Gaussian distribution.
        
        Inputs:
            out_mean (Tensor): Latent function mean, shape [B, C]
            out_var (Tensor): Latent function variance, shape [B, C] or [B, C, C]

        Outputs:
            posterior_predict_mean (Tensor): Predicted class probabilities, shape [B, C]
        """

        if out_var.dim() == 3:
            kappa = 1 / torch.sqrt(1. + np.pi / 8 * out_var.diagonal(dim1=1, dim2=2))
        else:
            kappa = 1 / torch.sqrt(1. + np.pi / 8 * out_var)
            
        posterior_predict_mean = torch.softmax(kappa * out_mean, dim=-1)
        return posterior_predict_mean

    def fit_scale_factor(self, data_loader, n_epoches, lr, speedup = True, verbose = False):
        """
        Fits the scale factor for predictive variance using negative log predictive density (NLPD).

        Inputs:
            data_loader (DataLoader): Dataloader containing (input, target) pairs
            n_epoches (int): Number of epochs for optimization
            lr (float): Learning rate for scale optimizer
            speedup (bool): If True (classification only), caches forward pass outputs to accelerate fitting
            verbose (bool): If True, prints NLPD at each epoch

        Outputs:
            total_train_nlpd (List[float]): Average NLPD per epoch over training data
        """
        print("fit scale factor")
        optimizer = torch.optim.Adam([self.scale_factor], lr)
        total_train_nlpd = []
        
        # store intermediate result and pack it into a data loader, so we only need to do one forward pass
        if speedup:
            
            if self.likelihood == 'regression':
                raise ValueError(f"Speed up not supported for regression atm")
            
            if self.likelihood == 'classification':

                f_mean = []
                f_var = []
                labels = []

                for (X, y) in tqdm(data_loader, desc= "packing f_mean f_var into a dataloader"):
                    out_mean, out_var = self.forward_latent(X.to(device))
                    f_mean.append(out_mean.detach().cpu().numpy())
                    f_var.append(out_var.detach().cpu().numpy())
                    if y.dim() == 2:
                        labels.append(y.numpy().argmax(1).reshape(-1, 1))
                    if y.dim() == 1:
                        labels.append(y.numpy().reshape(-1, 1))

                f_mean = np.vstack(f_mean)
                f_var = np.vstack(f_var)
                labels = np.vstack(labels)

                scale_fit_dataset = torch_dataset(f_mean, f_var, labels)
                scale_fit_dataloader = DataLoader(scale_fit_dataset, batch_size=16, shuffle=True)

                for epoch in tqdm(range(n_epoches), desc="fitting scaling factor"):
                    running_nlpd = 0
                    for data_pair in scale_fit_dataloader:
                        x_mean, x_var_label = data_pair
                        num_class = x_mean.shape[1]
                        x_mean = x_mean.to(device)
                        x_var, label = x_var_label.split(num_class, dim=1)
                        x_var = x_var.to(device)
                        label = label.to(device)

                        optimizer.zero_grad()
                        # make prediction
                        x_var = x_var / self.scale_factor.to(device)
                        posterior_predict_mean = self.probit_approximation(x_mean, x_var)
                        # construct log posterior predictive distribution
                        posterior_predictive_dist = Categorical(posterior_predict_mean)
                        # calculate nlpd and update
                        nlpd = -posterior_predictive_dist.log_prob(label).mean()
                        nlpd.backward()
                        optimizer.step()
                        # log nlpd
                        running_nlpd += nlpd.item()
                    total_train_nlpd.append(running_nlpd / len(scale_fit_dataloader))
                    if verbose:
                        print(f"epoch {epoch + 1}, nlpd {total_train_nlpd[-1]}")
                        
                del scale_fit_dataloader
                del scale_fit_dataset
        
        else:
            
            if self.likelihood == 'classification':
                for epoch in tqdm(range(n_epoches), desc="fitting scaling factor"):
                    running_nlpd = 0
                    for (data, label) in data_loader:
                        
                        data = data.to(device)
                        label = label.to(device)

                        optimizer.zero_grad()
                        # make prediction
                        posterior_predict_mean = self.forward(data)
                        # construct log posterior predictive distribution
                        posterior_predictive_dist = Categorical(posterior_predict_mean)
                        # calculate nlpd and update
                        nlpd = -posterior_predictive_dist.log_prob(label).mean()
                        nlpd.backward()
                        optimizer.step()
                        # log nlpd
                        running_nlpd += nlpd.item()
                    total_train_nlpd.append(running_nlpd / len(data_loader))
                    if verbose:
                        print(f"epoch {epoch + 1}, nlpd {total_train_nlpd[-1]}")


            if self.likelihood == 'regression':
                for epoch in tqdm(range(n_epoches), desc="fitting scaling factor"):
                    running_nlpd = 0
                    for (data, label) in data_loader:
                        data = data.to(device)
                        label = label.to(device)
                        
                        optimizer.zero_grad()
                        # make prediction
                        posterior_predict_mean, posterior_predict_var = self.forward(data)
                        # construct log posterior predictive distribution
                        posterior_predictive_dist = Normal(posterior_predict_mean, posterior_predict_var.sqrt())
                        # calculate nlpd and update
                        nlpd = -posterior_predictive_dist.log_prob(label).mean()
                        nlpd.backward()
                        optimizer.step()
                        # log nlpd
                        running_nlpd += nlpd.item()
                
                    total_train_nlpd.append(running_nlpd / len(data_loader))
                    
                    if verbose:
                        print(f"epoch {epoch + 1}, nlpd {total_train_nlpd[-1]}")
                    
        return total_train_nlpd