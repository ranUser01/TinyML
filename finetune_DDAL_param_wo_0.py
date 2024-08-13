from CNN_setup.model.CIFAR_CNN import CIFAR_CNN_Classifier
from CNN_setup.model.MNIST_CNN import Mnist_CNN_Classifier
from CNN_setup.utils.cnn_models_utils import load_model
from torchvision.transforms import  ToTensor, Compose, Grayscale
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder, MNIST
from DDAL_utils import DDAL_test, DDAL_test_gradual
import pickle
from itertools import product
    
## -------------- MNIST ------------------- ##
rotated = ImageFolder(root='data/transformed/mnist-rotated90', transform=Compose([ToTensor(),Grayscale(num_output_channels=1)]))
test_mnist = MNIST(root='./data', train=False, download=True, transform=ToTensor())

# lambidas = [x/20 for x in range(16, 20)]
# thetas = [x/20 for x in range(10, 20)]
lambidas = [x/40 for x in range(34, 40)]
thetas = [x/40 for x in range(30, 40)]
batch_sizes = [i*32 for i in range(1, 11)]

for la, th, bs in product(lambidas, thetas, batch_sizes):
    print("Lambda:", la, "Theta:", th, "Batch Size:", bs)
    
    orig_loader = DataLoader(test_mnist,  batch_size = bs, shuffle=True)
    drift_loader = DataLoader(dataset=rotated, batch_size = bs)
    
    model = load_model('trained_models/CNN_mnist_wo_0.torch', Mnist_CNN_Classifier())
    ## Sanity check to verify performence on clean test data 

    out = DDAL_test(orig_loader=orig_loader,drift_loader=None, model=model, size_batch = bs, theta = th, lambida = la)

    with open(f'experiments_results/finetune_ddal_wo_0/mnist_clean_test_la_{la}_th_{th}_bs_{bs}.dict', 'wb') as f:
        pickle.dump(out, f)
    
    print(out['Drift Detected'])
        
    ## Abrupt case withhold
        
    withhold_class = ImageFolder(root='data/transformed/mnist-w-0', transform=Compose([ToTensor(),Grayscale(num_output_channels=1)]))
    drift_loader = DataLoader(dataset=withhold_class, batch_size = bs)

    out = DDAL_test(orig_loader=orig_loader,drift_loader=drift_loader, model=model, size_batch = bs, theta = th, lambida = la)

    print(out['Drift Detected'])

    with open(f'experiments_results/finetune_ddal_wo_0/mnist_abrupt_w-0_la{la}_th{th}_bs{bs}.dict', 'wb') as f:
        pickle.dump(out, f)
        
    ## Gradual case withhold

    out = DDAL_test_gradual(orig_loader=orig_loader,drift_loader=drift_loader, model=model, size_batch = bs, theta = th, lambida = la)

    with open(f'experiments_results/finetune_ddal_wo_0/mnist_gradual_w-0_la{la}_th{th}_bs:{bs}.dict', 'wb') as f:
        pickle.dump(out, f)
        
    print(out['Drift Detected'])