import pandas as pd
from numpy import float32
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import relu

# Basic CNN 
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.linear1 = nn.Linear(784,250)
        self.linear2 = nn.Linear(250,100)
        self.linear3 = nn.Linear(100,10)

    def forward(self, x):
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        x = self.linear3(x)
        return x

# Custom dataset for image data frames, from any path given. label_col_name must be given 
class ImageDFDataset(Dataset):
    def __init__(self, path:str, label_col_name: str):
        df = pd.read_csv(path)
        self.images = df.drop(columns= label_col_name).to_numpy(float32)
        self.labels = df[label_col_name].to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx:int):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label
    

def unpickle(file:str):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def cifarToDf (path:str = "../data/cifar-10-python\data_batch_1"):
    cifar = unpickle(path)
    cifar_df = pd.DataFrame(cifar[b'data'])
    cifar_df['label'] = cifar[b'labels']

    return cifar_df