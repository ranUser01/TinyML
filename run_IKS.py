from torchvision.transforms import  ToTensor, Compose, Grayscale
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder, MNIST
from IKS_utils import test_IKS_gradual, test_IKS_abrupt
import pickle

bs = 1 ## batch size

# import time  // add another argument to the experiment files to store more than 1 run
# cases = ['cifar', 'mnist']
    
## -------------- MNIST ------------------- ##
rotated = ImageFolder(root='data/transformed/mnist-rotated90', transform=Compose([ToTensor(),Grayscale(num_output_channels=1)]))
drift_loader = DataLoader(dataset=rotated, batch_size = bs)
test_mnist = MNIST(root='./data', train=False, download=True, transform=ToTensor())
orig_loader = DataLoader(test_mnist, batch_size = 1, shuffle=True)

# ## Sanity check to verify performence on clean test data 

out = test_IKS_abrupt(orig_loader=orig_loader,drift_loader=None)

with open('experiments_results/IKS/mnist_clean_test.dict', 'wb') as f:
    pickle.dump(out, f)
    
print(out['Drift Detected'])
    
## Abrupt case withhold
    
withhold_class = ImageFolder(root='data/transformed/mnist-w-0', transform=Compose([ToTensor(),Grayscale(num_output_channels=1)]))
drift_loader = DataLoader(dataset=withhold_class, batch_size = bs)

out = test_IKS_abrupt(orig_loader=orig_loader,drift_loader=drift_loader)

with open('experiments_results/IKS/mnist_abrupt_w-0.dict', 'wb') as f:
    pickle.dump(out, f)
    
print(out['Drift Detected'])
    
## Gradual case withhold

out = test_IKS_gradual(orig_loader=orig_loader,drift_loader=drift_loader)

with open('experiments_results/IKS/mnist_gradual_w-0.dict', 'wb') as f:
    pickle.dump(out, f)
    
print(out['Drift Detected'])
    
# ## Abrupt case rotation

out = test_IKS_abrupt(orig_loader=orig_loader,drift_loader=drift_loader)

with open('experiments_results/IKS/mnist_rotate_abrupt_rotate.dict', 'wb') as f:
    pickle.dump(out, f)
    
print(out['Drift Detected'])

## Gradual case rotation

out = test_IKS_gradual(orig_loader=orig_loader,drift_loader=drift_loader)

with open('experiments_results/IKS/mnist_rotate_gradual_rotate.dict', 'wb') as f:
    pickle.dump(out, f)
    
print(out['Drift Detected'])

## -------------- CIFAR ------------------- ##
rotated = ImageFolder(root='data/transformed/cifar-rotated90', transform=ToTensor())
drift_loader = DataLoader(dataset=rotated, batch_size = bs)
test_cifar = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())
orig_loader = DataLoader(test_cifar,  batch_size = bs, shuffle=True)

out = test_IKS_abrupt(orig_loader=orig_loader,drift_loader=drift_loader)

with open('experiments_results/IKS/cifar_rotate_abrupt_rotate.dict', 'wb') as f:
    pickle.dump(out, f)
    
print(out['Drift Detected'])

out = test_IKS_gradual(orig_loader=orig_loader,drift_loader=drift_loader)

with open('experiments_results/IKS/cifar_rotate_gradual_rotate.dict', 'wb') as f:
    pickle.dump(out, f)
    
print(out['Drift Detected'])
    
withhold_class = ImageFolder(root='data/transformed/cifar-w-0', transform=ToTensor())
drift_loader = DataLoader(dataset=withhold_class, batch_size = bs)

out = test_IKS_abrupt(orig_loader=orig_loader,drift_loader=drift_loader)
    
print(out['Drift Detected'])

with open('experiments_results/IKS/cifar_abrupt_w-0.dict', 'wb') as f:
    pickle.dump(out, f)

out = test_IKS_gradual(orig_loader=orig_loader,drift_loader=drift_loader)
    
print(out['Drift Detected'])

with open('experiments_results/IKS/cifar_gradual_w-0.dict', 'wb') as f:
    pickle.dump(out, f)