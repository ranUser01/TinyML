from CNN_setup.model.CIFAR_CNN import CIFAR_CNN_Classifier
from CNN_setup.model.MNIST_CNN import Mnist_CNN_Classifier
from CNN_setup.utils.cnn_models_utils import load_model
from torchvision.transforms import  ToTensor, Compose, Grayscale
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder, MNIST
from DDAL_utils import DDAL_test, DDAL_test_gradual
import pickle

bs = 32 ## batch size

# import time  // add another argument to the experiment files to store more than 1 run
# cases = ['cifar', 'mnist']
    
## -------------- MNIST ------------------- ##
rotated = ImageFolder(root='data/transformed/mnist-rotated90', transform=Compose([ToTensor(),Grayscale(num_output_channels=1)]))
drift_loader = DataLoader(dataset=rotated, batch_size = bs)
test_mnist = MNIST(root='./data', train=False, download=True, transform=ToTensor())
orig_loader = DataLoader(test_mnist,  batch_size=32, shuffle=True)

model = load_model('trained_models/CNN_mnist_wo_0.torch', Mnist_CNN_Classifier())
##IT LOOKS LIKE THAT WHEN A MODEL IS TRAINED TO DETECT 0 THEN IT WILL NOT PERFORM WELL TO DETECT THOSE AS IT WILL BE CONFIDENT AS TO
## WHAT THEIR CLASS SHOULD BE. NOTE TREAIN A MODEL WITHOUT ZEROS AND COMPARE ##

# ## Sanity check to verify performence on clean test data 

out = DDAL_test(orig_loader=orig_loader,drift_loader=None, model=model, size_batch = bs,  theta  = 0.750, lambida = 0.850)

with open('experiments_results/DDAL_wo_model/mnist_clean_test.dict', 'wb') as f:
    pickle.dump(out, f)
    
print(out['Drift Detected'])
    
## Abrupt case withhold
    
withhold_class = ImageFolder(root='data/transformed/mnist-w-0', transform=Compose([ToTensor(),Grayscale(num_output_channels=1)]))
drift_loader = DataLoader(dataset=withhold_class, batch_size = bs)

out = DDAL_test(orig_loader=orig_loader,drift_loader=drift_loader, model=model, size_batch = bs,  theta  = 0.750, lambida = 0.850)

print(out['Drift Detected'])

with open('experiments_results/DDAL_wo_model/mnist_abrupt_w-0.dict', 'wb') as f:
    pickle.dump(out, f)
    
## Gradual case withhold

out = DDAL_test_gradual(orig_loader=orig_loader,drift_loader=drift_loader, model=model, size_batch = bs,  theta  = 0.750, lambida = 0.850)

with open('experiments_results/DDAL_wo_model/mnist_gradual_w-0.dict', 'wb') as f:
    pickle.dump(out, f)
    
print(out['Drift Detected'])
    
# ## Abrupt case rotation

out = DDAL_test(orig_loader=orig_loader,drift_loader=drift_loader, model=model)

with open('experiments_results/DDAL_wo_model/mnist_rotate_abrupt_rotate.dict', 'wb') as f:
    pickle.dump(out, f)
    
print(out['Drift Detected'])

## Gradual case rotation

out = DDAL_test_gradual(orig_loader=orig_loader,drift_loader=drift_loader, model=model)

with open('experiments_results/DDAL_wo_model/mnist_rotate_gradual_rotate.dict', 'wb') as f:
    pickle.dump(out, f)
    
print(out['Drift Detected'])
    