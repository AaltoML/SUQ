from transformers import ViTForImageClassification, AutoImageProcessor
from transformers import get_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import ivon
import itertools

from suq import streamline_vit

from prepare_dataset import classification_vit_dataloader
from helper_function import eval_vit_performance
from vit_model import ViT_Classification, GPTConfig

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--load_checkpoint", type=int, default = 0, help="whether train from scratch")
parser.add_argument("--hyperparameter_id", type=int, default = 0, help="id for hyperparameter")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##################################### set hyperparameters #####################################
finetune_attn = 1
num_det_blocks = 10
batch_size = 16
dataset_name, num_classes, N_data ='dtd', 47, 1880
num_epochs = 50
n_samples = 50

lr_values = [0.5, 0.1, 0.05, 0.01, 0.005]
h_0_values = [0.01, 0.05, 0.1]

hyperparameter_grid = list(itertools.product(lr_values, h_0_values))
lr, h_0 = hyperparameter_grid[args.hyperparameter_id]
##################################### load model #####################################
model_name = "google/vit-base-patch16-224"
huggingface_model = ViTForImageClassification.from_pretrained(model_name)
image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

## init model
config = GPTConfig('small')
model = ViT_Classification(config, huggingface_model, num_classes, return_logits = True)
# hugging face default vit flash attention false
for block in model.transformer.h:
    block.attn.flash = False

model = model.to(device)

## freeze all parameters except classifier and mlp/attn in the corresponding transformer blocks
for param in model.transformer.parameters():
    param.requires_grad = False

params_to_optimize = list(model.classifier.parameters())
for block_index in range(len(model.transformer.h)):
    if block_index >= num_det_blocks:
        if finetune_attn:
            print("fine tune attention value only")
            for name, param in model.transformer.h[block_index].attn.named_parameters():
                if 'c_attn_v.weight' in name:
                    param.requires_grad = True
                params_to_optimize.extend([param for name, param in model.transformer.h[block_index].attn.named_parameters() if 'c_attn_v.weight' in name])
        else:
            print("fine tune mlp only")
            for name, param in model.transformer.h[block_index].mlp.named_parameters():
                if 'c_fc.weight' in name:
                    param.requires_grad = True
                
                if 'c_proj.weight' in name:
                    param.requires_grad = True
                
                params_to_optimize.extend([param for name, param in model.transformer.h[block_index].mlp.named_parameters() if 'weight' in name])

for param in model.classifier.parameters():
    param.requires_grad = True

##################################### dataset loader #####################################
train_loader, test_loader, val_loader = classification_vit_dataloader(dataset_name, batch_size, image_processor)

##################################### training  #####################################
# init IVON optimiser
weight_decay = 1e-5
linear_head_optimizer = optim.Adam(model.classifier.parameters())
optimizer = ivon.IVON(params_to_optimize, lr=lr, ess=N_data, weight_decay=weight_decay, hess_init=h_0)

if args.load_checkpoint:
    if finetune_attn:
        checkpoint = torch.load(f"{dataset_name}_vit_ivon_attn.pt")
        opt_checkpoint = torch.load(f"{dataset_name}_vit_ivon-posterior_attn.pth")
        model.load_state_dict(checkpoint)
        optimizer.load_state_dict(opt_checkpoint)
    else:
        checkpoint = torch.load(f"{dataset_name}_vit_ivon_mlp.pt")
        opt_checkpoint = torch.load(f"{dataset_name}_vit_ivon-posterior_mlp.pth")
        model.load_state_dict(checkpoint)
        optimizer.load_state_dict(opt_checkpoint)
else:
    criterion = nn.CrossEntropyLoss()
    train_samples = 1
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0.1 * num_training_steps,  num_training_steps=num_training_steps) 

    best_val_acc = 0

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        
        # Train 
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader: 
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            if epoch == 0:
                linear_head_optimizer.zero_grad()
                
                pred = model(inputs)
                loss = criterion(pred, labels)
                loss.backward()
                linear_head_optimizer.step()
                
            else:
                optimizer.zero_grad()
                
                for _ in range(train_samples):
                    with optimizer.sampled_params(train=True):
                        pred = model(inputs)
                        loss = criterion(pred, labels)
                        loss.backward()
                    
                    optimizer.step()
                    lr_scheduler.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"    Training Loss: {epoch_loss:.4f}")
        train_accuracy = eval_vit_performance(model, train_loader, device)
        print(f"    Train Accuracy: {train_accuracy:.3f}")
        val_accuracy = eval_vit_performance(model, val_loader, device, full_dataset=True)  
        print(f"    Val Accuracy: {val_accuracy:.3f}")
        
        if epoch != 0:
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                if finetune_attn:
                    torch.save(model.state_dict(), f'{dataset_name}_vit_ivon_attn_{args.hyperparameter_id}.pt')
                    torch.save(optimizer.state_dict(), f"{dataset_name}_vit_ivon-posterior_attn_{args.hyperparameter_id}.pth")
                else:
                    torch.save(model.state_dict(), f'{dataset_name}_vit_ivon_mlp_{args.hyperparameter_id}.pt')
                    torch.save(optimizer.state_dict(), f"{dataset_name}_vit_ivon-posterior_mlp_{args.hyperparameter_id}.pth")

model.eval()
softmax = nn.Softmax()
total_acc = []
with torch.no_grad():
    for X, y in tqdm(test_loader, desc = "MAP Evaluating"):
        pred = model(X.to(device))
        pred = softmax(pred) # MAP model needs softmax
        label = y.to(device)
        acc = (pred.argmax(1) == label.argmax(1)).float().cpu()
        total_acc.extend(acc)

print(f"MAP test accuracy {np.mean(total_acc):.3f}")

##################################### Eval IVON #####################################
total_acc = []
with torch.no_grad():
    for X, y in tqdm(test_loader, desc = "IVON Evaluating"):
        
        samples = torch.zeros((n_samples, X.shape[0], num_classes), device = X.device)
        for i in range(n_samples):
            with optimizer.sampled_params(train=False):
                pred = model(X)
                pred = softmax(pred) # IVON model needs softmax
            samples[i] = pred.squeeze().detach()
        pred = torch.mean(samples, axis=0)

        label = y.to(device)
        acc = (pred.argmax(1) == label.argmax(1)).float().cpu()
        total_acc.extend(acc)

print(f"IVON test accuracy {np.mean(total_acc):.3f}")

##################################### init our model and fit scale factor #####################################
# init our model
if finetune_attn:
    MLP_determinstic = True
    Attn_determinstic = False
else:
    MLP_determinstic = False
    Attn_determinstic = True

posterior_variance = 1 / (optimizer.param_groups[0]['ess'] * (optimizer.param_groups[0]['hess'] + optimizer.param_groups[0]['weight_decay']))

scale_init, scale_fit_epoch, scale_fit_lr  = 1, 50, 1e-3
suq_model = streamline_vit(model = model, 
                           posterior = posterior_variance,
                           covariance_structure = 'diag',
                           likelihood = 'classification',
                           MLP_deterministic = MLP_determinstic,
                           Attn_deterministic = Attn_determinstic, 
                           attention_diag_cov = False,
                           num_det_blocks = num_det_blocks,
                           scale_init = scale_init)

suq_model.fit_scale_factor(train_loader, scale_fit_epoch, scale_fit_lr)

total_acc = []
for X, y in tqdm(test_loader, desc = "SUQ Evaluating"):
    pred = suq_model(X.to(device))
    label = y.to(device)
    acc = (pred.argmax(1) == label.argmax(1)).float().cpu()
    total_acc.extend(acc)

print(f"SUQ test accuracy {np.mean(total_acc):.3f}")
