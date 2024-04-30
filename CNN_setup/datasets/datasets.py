from torch import manual_seed
from torchvision.datasets import CIFAR10
from torchvision.datasets.mnist import MNIST 
from PIL import Image

class CustomCIFAR10(CIFAR10):
    def __init__(self, root, train=False, transform=None, target_transform=None, download=False, manual_seed=None):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.manual_seed = manual_seed

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        
        if self.manual_seed is not None:
            manual_seed(self.manual_seed)

        if self.transform is not None:
            img = self.transform(img)   

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    
class CustomMNIST(MNIST):
    def __init__(self, root, train=False, transform=None, target_transform=None, download=False, manual_seed=None):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.manual_seed = manual_seed

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        
        if self.manual_seed is not None:
            manual_seed(self.manual_seed)
            
        if self.transform is not None:
            img = self.transform(img)   
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target