##################################### load stuff #####################################
import torch
import torch.nn as nn
import numpy as np
from laplace import Laplace
from prepare_dataset import load_mnist
from tqdm import tqdm
from suq import streamline_mlp
from helper_function import create_mlp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = './'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--load_checkpoint", type=int, default = 0, help="whether train from scratch")
args = parser.parse_args()

##################################### set hyperparameters #####################################
batch_size, lr, weight_decay, n_epoch, network_structure = [64, 1e-3, 1e-5, 15, [784, 128, 64, 10]]
##################################### dataset loader #####################################
train_loader, test_loader = load_mnist(batch_size, data_dir)
##################################### define model #####################################
model = create_mlp(network_structure, 'relu', 'classification')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_func = nn.CrossEntropyLoss()

##################################### train model #####################################
if args.load_checkpoint:
    checkpoint = torch.load(f"mnist_mlp.pt")
    model.load_state_dict(checkpoint)
else:
    for epoch in tqdm(range(n_epoch), desc = "Training"):
        train_loss = []

        for X, y in train_loader:
            optimizer.zero_grad()
            loss = loss_func(model(X.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())

    torch.save(model.state_dict(), f'mnist_mlp.pt')

model.eval()
total_acc = []
with torch.no_grad():
    for X, y in tqdm(test_loader, desc = "MAP Evaluating"):
        pred = model(X.to(device))
        label = y.to(device)
        acc = (pred.argmax(1) == label).float().cpu()
        total_acc.extend(acc)

print(f"MAP test accuracy {np.mean(total_acc):.3f}")
##################################### fit laplace to fully connected layers #####################################
if args.load_checkpoint:
    la = torch.load("mnist_mlp_la.pt")
else:
    ### define lapalce
    la = Laplace(model[:-1], 'classification', subset_of_weights='all', hessian_structure='diag')
    ### learn Hessian
    la.fit(train_loader)
    ### learn prior precision
    la.optimize_prior_precision(
        method="marglik",
        pred_type="glm",
        link_approx="probit",
        val_loader=train_loader
    )

    torch.save(la, "mnist_mlp_la.pt")

total_acc = []
with torch.no_grad():
    for X, y in tqdm(test_loader, desc = "LA Evaluating"):
        pred = la(X.to(device))
        label = y.to(device)
        acc = (pred.argmax(1) == label).float().cpu()
        total_acc.extend(acc)

print(f"LA test accuracy {np.mean(total_acc):.3f}")
##################################### make prediction with DBNN #####################################
scale_init = 1.0
suq_model = streamline_mlp(model = model[:-1], 
                           posterior = la.posterior_variance.detach(), 
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


