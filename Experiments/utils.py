import pandas as pd
import numpy as np 
from torch.utils.data import Dataset

from pathlib import Path
from typing import Union

class ImageDFDataset(Dataset):
    '''Custom dataset for image data frames, from any path given. label_col_name must be given '''
    
    def __init__(self, path:Union[str, Path], label_col_name: str):
        df = pd.read_csv(path)
        self.images = df.drop(columns= label_col_name).to_numpy(np.float32)
        self.labels = df[label_col_name].to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx:int):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label
    

def unpickle(file:Union[str, Path]):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def cifarToDf (path:Union[str, Path] = "../data/cifar-10-python/data_batch_1"):
    cifar = unpickle(path)
    cifar_df = pd.DataFrame(cifar[b'data'])
    cifar_df['label'] = cifar[b'labels']

    return cifar_df
