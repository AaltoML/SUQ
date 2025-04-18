import torch
from tqdm import tqdm

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