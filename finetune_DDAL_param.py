from CNN_setup.model.CIFAR_CNN import CIFAR_CNN_Classifier
from CNN_setup.model.MNIST_CNN import Mnist_CNN_Classifier
from CNN_setup.utils.cnn_models_utils import load_model
from torchvision.transforms import  ToTensor, Compose, Grayscale
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder, MNIST
from DDAL_utils import DDAL_test, DDAL_test_gradual
import pickle

# import time  // add another argument to the experiment files to store more than 1 run
# cases = ['cifar', 'mnist']

# ## -------------- CIFAR ------------------- ##
# rotated = ImageFolder(root='data/transformed/cifar-rotated90', transform=ToTensor())
# drift_loader = DataLoader(dataset=rotated, batch_size = 32)
# test_cifar = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())
# orig_loader = DataLoader(test_cifar,  batch_size=32, shuffle=True)

# model = load_model('trained_models/CNN_cifar_downloaded.torch', CIFAR_CNN_Classifier())

# out = DDAL_test(orig_loader=orig_loader,drift_loader=drift_loader, model=model)

# with open('experiments_results/cifar_rotate_abrupt_rotate.dict', 'wb') as f:
#     pickle.dump(out, f)

# out = DDAL_test_gradual(orig_loader=orig_loader,drift_loader=drift_loader, model=model)

# with open('experiments_results/cifar_rotate_gradual_rotate.dict', 'wb') as f:
#     pickle.dump(out, f)
    
# withhold_class = ImageFolder(root='data/transformed/cifar-w-0', transform=ToTensor())
# drift_loader = DataLoader(dataset=withhold_class, batch_size = 32)

# out = DDAL_test(orig_loader=orig_loader,drift_loader=drift_loader, model=model)

# with open('experiments_results/cifar_abrupt_w-0.dict', 'wb') as f:
#     pickle.dump(out, f)

# out = DDAL_test_gradual(orig_loader=orig_loader,drift_loader=drift_loader, model=model)

# with open('experiments_results/cifar_gradual_w-0.dict', 'wb') as f:
#     pickle.dump(out, f)
    

## -------------- MNIST ------------------- ##
rotated = ImageFolder(root='data/transformed/mnist-rotated90', transform=Compose([ToTensor(),Grayscale(num_output_channels=1)]))
drift_loader = DataLoader(dataset=rotated, batch_size = 32)
test_mnist = MNIST(root='./data', train=False, download=True, transform=ToTensor())
orig_loader = DataLoader(test_mnist,  batch_size=32, shuffle=True)

lambidas = [0.05,0.2,0.5,0.8,0.9,0.95] + [x for x in reversed([0.05,0.2,0.5,0.8,0.9,0.95])]
thetas =  [x for x in reversed([0.05,0.2,0.5,0.8,0.9,0.95])] + [0.05,0.2,0.5,0.8,0.9,0.95]

for la, th in zip(lambidas, thetas):
    print(la,th)
    model = load_model('trained_models/CNN_mnist_downloaded.torch', Mnist_CNN_Classifier())
    ## Sanity check to verify performence on clean test data 

    out = DDAL_test(orig_loader=orig_loader,drift_loader=None, model=model, size_batch = 32, theta = th, lambida = la)

    # with open('experiments_results/mnist_clean_test.dict', 'wb') as f:
    #     pickle.dump(out, f)
    
    print(out['Drift Detected'])
        
    ## Abrupt case withhold
        
    withhold_class = ImageFolder(root='data/transformed/mnist-w-0', transform=Compose([ToTensor(),Grayscale(num_output_channels=1)]))
    drift_loader = DataLoader(dataset=withhold_class, batch_size = 32)

    out = DDAL_test(orig_loader=orig_loader,drift_loader=drift_loader, model=model, size_batch = 32, theta = th, lambida = la)

    print(out['Drift Detected'])

    # with open('experiments_results/mnist_abrupt_w-0.dict', 'wb') as f:
    #     pickle.dump(out, f)
        
    ## Gradual case withhold

    out = DDAL_test_gradual(orig_loader=orig_loader,drift_loader=drift_loader, model=model, size_batch = 32, theta = th, lambida = la)

    # with open('experiments_results/mnist_gradual_w-0.dict', 'wb') as f:
    #     pickle.dump(out, f)
        
    print(out['Drift Detected'])
    



