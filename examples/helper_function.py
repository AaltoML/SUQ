import torch
from tqdm import tqdm
import torch.nn as nn

def create_mlp(layer_list, act_func, task):
    
    model = nn.Sequential()
    
    for i in range(len(layer_list) - 1):
        model.append(nn.Linear(layer_list[i], layer_list[i+1]))
        if act_func == 'relu':
            model.append(nn.ReLU())
        
        if act_func == 'tanh':
            model.append(nn.Tanh())
    
    if task == 'classification':
        # replace last activation into softmax
        model[-1] = nn.Softmax()
    
    if task == 'regression':
        # no explicit activation for regression
        model = model[:-1]
    
    return model

def eval_vit_performance(model, data_loader, device, full_dataset = False):
    model.eval()
    
    running_acc = 0
    with torch.no_grad():
        if full_dataset:
            for (inputs, labels) in tqdm(data_loader, desc="Evaluating"):
                inputs, labels = inputs.to(device), labels.to(device).float()
                
                pred = model(inputs)
                running_acc += (pred.argmax(1) == labels.argmax(1)).float().mean().item()
            
            accuracy = running_acc / len(data_loader)
        else:
            for i in range(20):
                inputs, labels =  next(iter(data_loader))
                inputs, labels = inputs.to(device), labels.to(device).float()
                
                pred = model(inputs)
                running_acc += (pred.argmax(1) == labels.argmax(1)).float().mean().item()
                
            accuracy = running_acc / 20
    
    return accuracy