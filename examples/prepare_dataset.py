
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from datasets import load_dataset


cache_dir = "./"

##################################### MLP Classification #####################################

def load_mnist(batch_size, data_dir):

    img_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,)),
                                            transforms.Lambda(lambda x: x.view(-1)),]) 
    target_transform = transforms.Compose([lambda y: torch.LongTensor([y]),
                                        lambda y: y.squeeze()])



    train_set = torchvision.datasets.MNIST(root = data_dir, train=True, transform=img_transform, target_transform=target_transform, download=True)
    test_set = torchvision.datasets.MNIST(root = data_dir, train=False, transform=img_transform, target_transform=target_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader

##################################### ViT Classification #####################################

class torch_dataset_from_hf(Dataset):
    def __init__(self, dataset, dataset_name, img_processor):
        
        self.image_processor = img_processor
        
        if dataset_name == 'cifar-10':
            self.X = dataset['img']
            self.Y = dataset['label']
            self.num_classes = 10
        
        if dataset_name == 'cifar-100':
            self.X = dataset['img']
            self.Y = dataset['fine_label']
            self.num_classes = 100
            
        if dataset_name == 'resisc':
            self.X = dataset['image']
            self.Y = dataset['label']
            self.num_classes = 45
            
        if dataset_name == 'dtd':
            self.X = dataset['image']
            self.Y = dataset['label']
            self.num_classes = 47
        
        if dataset_name == 'eurosat':
            self.X = dataset['image']
            self.Y = dataset['label']
            self.num_classes = 10
        
        if dataset_name == 'svhn':
            self.X = dataset['image']
            self.Y = dataset['label']
            self.num_classes = 10   
    
        if dataset_name == 'imagenet-r':
            self.X = dataset['image']
            self.Y = dataset['labels']
            self.num_classes = 100
    
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        img, label = self.X[idx], self.Y[idx]
        
        if img.mode != 'RGB':
            img_pt = self.image_processor(img.convert('RGB'), return_tensors="pt")['pixel_values'].squeeze()
        else:
            img_pt = self.image_processor(img, return_tensors="pt")['pixel_values'].squeeze()
        label_pt = F.one_hot(torch.Tensor([label]).long(), num_classes=self.num_classes).float().squeeze()
        
        dtype = torch.get_default_dtype()
        return img_pt.type(dtype), label_pt

def classification_vit_dataloader(dataset_name, batch_size, image_processer):
    
    if dataset_name == 'cifar-10':
        ds = load_dataset("uoft-cs/cifar10", cache_dir=cache_dir)
    
    if dataset_name == 'cifar-100':
        ds = load_dataset("uoft-cs/cifar100", cache_dir=cache_dir)
    
    if dataset_name == 'dtd':
        ds = load_dataset("jxie/dtd", cache_dir=cache_dir)
    
    if dataset_name == 'resisc':
        ds = load_dataset("timm/resisc45", cache_dir=cache_dir)
    
    if dataset_name == 'eurosat':
        ds = load_dataset("blanchon/EuroSAT_RGB", cache_dir=cache_dir)
    
    if dataset_name == 'svhn':
        ds = load_dataset("ufldl-stanford/svhn", "cropped_digits", cache_dir=cache_dir)
        
    if dataset_name == 'imagenet-r':
        ds = load_dataset("axiong/imagenet-r", cache_dir=cache_dir)
        class_names = list(set(ds['test']['class_name']))
        class_names.sort()
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

        def convert_class_name_to_idx(example):
            example['labels'] = class_to_idx[example['class_name']]
            return example

        ds = ds.map(convert_class_name_to_idx, batched=False)
        class_indices = list(range(len(class_names)))
        np.random.seed(42)
        sampled_class_indices = np.random.choice(class_indices, size=100, replace=False).tolist()
        sampled_class_names = [class_names[idx] for idx in sampled_class_indices]

        new_class_to_idx = {class_name: new_idx for new_idx, class_name in enumerate(sampled_class_names)}

        def remap_labels(example):
            if class_names[example['labels']] in new_class_to_idx:
                example['labels'] = new_class_to_idx[class_names[example['labels']]]
            return example

        filtered_dataset = ds.filter(lambda example: class_names[example['labels']] in sampled_class_names)
        remapped_dataset = filtered_dataset.map(remap_labels, batched=False)

        shuffled_dataset = remapped_dataset['test'].shuffle(seed=42)

        train_test_valid_split = shuffled_dataset.train_test_split(test_size=0.2, seed=42)
        test_valid_split = train_test_valid_split['test'].train_test_split(test_size=0.5, seed=42)

        ds = {'train': train_test_valid_split['train'],
                'validation': test_valid_split['train'],
                'test': test_valid_split['test']}
    
    test_set = torch_dataset_from_hf(ds['test'], dataset_name, image_processer)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    try:
        val_set = torch_dataset_from_hf(ds['validation'], dataset_name, image_processer)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
        
        train_set = torch_dataset_from_hf(ds['train'], dataset_name, image_processer)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    except:
        print(f"no validation set for {dataset_name}, split training set to create one")

        split_dataset = ds['train'].train_test_split(test_size=0.2, seed=42)

        val_set = torch_dataset_from_hf(split_dataset['test'], dataset_name, image_processer)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
        
        train_set = torch_dataset_from_hf(split_dataset['train'], dataset_name, image_processer)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        
    return train_loader, test_loader, val_loader