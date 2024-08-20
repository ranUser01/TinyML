from torchvision.transforms import  ToTensor, Compose, Grayscale
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder, MNIST
from IKS_utils import test_IKS_gradual, test_IKS_abrupt
import pickle

bs = 32 ## batch size
    
## -------------- MNIST ------------------- ##
rotated = ImageFolder(root='data/transformed/mnist-rotated90', transform=Compose([ToTensor(),Grayscale(num_output_channels=1)]))
drift_loader = DataLoader(dataset=rotated, batch_size = bs)
test_mnist = MNIST(root='./data', train=False, download=True, transform=ToTensor())
orig_loader = DataLoader(test_mnist, batch_size = bs, shuffle=True)


out = test_IKS_abrupt(orig_loader=orig_loader,drift_loader=None, dataset_shape = 28)


# ## Abrupt case rotation

out = test_IKS_abrupt(orig_loader=orig_loader,drift_loader=drift_loader, dataset_shape = 28)

## Gradual case rotation

out = test_IKS_gradual(orig_loader=orig_loader,drift_loader=drift_loader, dataset_shape = 28)
    
## Abrupt case withhold
    
withhold_class = ImageFolder(root='data/transformed/mnist-w-0', transform=Compose([ToTensor(),Grayscale(num_output_channels=1)]))
drift_loader = DataLoader(dataset=withhold_class, batch_size = bs)

out = test_IKS_abrupt(orig_loader=orig_loader,drift_loader=drift_loader, dataset_shape = 28)

    
# Gradual case withhold

out = test_IKS_gradual(orig_loader=orig_loader,drift_loader=drift_loader, dataset_shape = 28)



## -------------- CIFAR ------------------- ##
rotated = ImageFolder(root='data/transformed/cifar-rotated90', transform=ToTensor())
drift_loader = DataLoader(dataset=rotated, batch_size = bs)
test_cifar = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())
orig_loader = DataLoader(test_cifar,  batch_size = bs, shuffle=True)

## Sanity check to verify performence on clean test data 

out = test_IKS_abrupt(orig_loader=orig_loader,drift_loader=None, dataset_shape = 28)

out = test_IKS_abrupt(orig_loader=orig_loader,drift_loader=drift_loader, dataset_shape = 32)


out = test_IKS_gradual(orig_loader=orig_loader,drift_loader=drift_loader, dataset_shape = 32)

    
withhold_class = ImageFolder(root='data/transformed/cifar-w-0', transform=ToTensor())
drift_loader = DataLoader(dataset=withhold_class, batch_size = bs)

out = test_IKS_abrupt(orig_loader=orig_loader,drift_loader=drift_loader, dataset_shape = 32)
    

out = test_IKS_gradual(orig_loader=orig_loader,drift_loader=drift_loader, dataset_shape = 32)
    