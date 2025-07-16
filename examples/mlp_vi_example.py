##################################### load stuff #####################################
import torch
import torch.nn as nn
import numpy as np
from prepare_dataset import load_mnist
from tqdm import tqdm
from suq import streamline_mlp
import ivon
from helper_function import create_mlp
from transformers import get_cosine_schedule_with_warmup

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = './'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--load_checkpoint", type=int, default = 0, help="whether train from scratch")
args = parser.parse_args()

##################################### set hyperparameters #####################################
batch_size, lr, weight_decay, n_epoch, network_structure = [64, 1e-3, 1e-5, 15, [784, 128, 64, 10]]
n_samples, num_classes = 1000, 10
lr, h_0 = 0.01, 0.1
best_val_acc = 0
##################################### dataset loader #####################################
train_loader, test_loader = load_mnist(batch_size, data_dir)
N_data = len(train_loader.dataset)
##################################### define model #####################################
ivon_model = create_mlp(network_structure, 'relu', 'classification')
ivon_model.to(device)
train_samples = 1
optimizer = ivon.IVON(ivon_model.parameters(), lr=lr, ess=N_data, weight_decay=weight_decay, hess_init=h_0)
loss_func = nn.CrossEntropyLoss()
num_training_steps = n_epoch * len(train_loader)
lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0,  num_training_steps=num_training_steps)
##################################### train model #####################################
if args.load_checkpoint:
    checkpoint = torch.load(f"mnist_mlp_ivon.pth")
    opt_checkpoint = torch.load(f"mnist_mlp_ivon-posterior.pth")
    ivon_model.load_state_dict(checkpoint)
    optimizer.load_state_dict(opt_checkpoint)
else:
    for epoch in tqdm(range(n_epoch), desc = "Training"):
        train_loss = []
        ivon_model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            for _ in range(train_samples):
                with optimizer.sampled_params(train=True):
                    pred = ivon_model[:-1](X)
                    loss = loss_func(pred, y)
                    loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            
            train_loss.append(loss.item())
            
    torch.save(ivon_model.state_dict(), f"mnist_mlp_ivon.pth")
    torch.save(optimizer.state_dict(), f"mnist_mlp_ivon-posterior.pth")

ivon_model.eval()
total_acc = []
with torch.no_grad():
    for X, y in tqdm(test_loader, desc = "MAP Evaluating"):
        pred = ivon_model(X.to(device))
        label = y.to(device)
        acc = (pred.argmax(1) == label).float().cpu()
        total_acc.extend(acc)

print(f"MAP test accuracy {np.mean(total_acc):.3f}")
##################################### eval IVON #####################################
total_acc = []
with torch.no_grad():
    for X, y in tqdm(test_loader, desc = "IVON Evaluating"):
        
        samples = torch.zeros((n_samples, X.shape[0], num_classes), device = X.device)
        for i in range(n_samples):
            with optimizer.sampled_params(train=False):
                pred = ivon_model(X)
            samples[i] = pred.squeeze().detach()
        pred = torch.mean(samples, axis=0)

        label = y.to(device)
        acc = (pred.argmax(1) == label.argmax(1)).float().cpu()
        total_acc.extend(acc)

print(f"IVON test accuracy {np.mean(total_acc):.3f}")
##################################### make prediction with DBNN #####################################
posterior_variance = 1 / (optimizer.param_groups[0]['ess'] * (optimizer.param_groups[0]['hess'] + optimizer.param_groups[0]['weight_decay']))

scale_init = 1.0
suq_model = streamline_mlp(model = ivon_model[:-1], 
                           posterior = posterior_variance, 
                           covariance_structure = 'diag', 
                           likelihood = 'classification', 
                           scale_init = scale_init)

suq_model.fit_scale_factor(train_loader, 10, 1e-5)

total_acc = []
for X, y in tqdm(test_loader, desc = "SUQ Evaluating"):
    pred = suq_model(X.to(device))
    label = y.to(device)
    acc = (pred.argmax(1) == label).float().cpu()
    total_acc.extend(acc)

print(f"SUQ test accuracy {np.mean(total_acc):.3f}")


