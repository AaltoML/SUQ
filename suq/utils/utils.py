from torch.utils.data import Dataset
import torch

class torch_dataset(Dataset):
    def __init__(self, x_data, y_data, z_data):

        self.X = torch.from_numpy(x_data).float()
        self.Y = torch.hstack([torch.from_numpy(y_data).float(), torch.from_numpy(z_data).float()])

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]