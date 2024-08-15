from CNN_setup.model.CIFAR_CNN import CIFAR_CNN_Classifier
from CNN_setup.model.MNIST_CNN import Mnist_CNN_Classifier
from CNN_setup.utils.cnn_models_utils import load_model
from torchvision.transforms import  ToTensor, Compose, Grayscale
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder, MNIST
from DDAL_utils import DDAL_test, DDAL_test_gradual
import pickle
from itertools import product

lambidas = [x/40 for x in range(32, 40)]
thetas = [x/40 for x in range(32, 40)]
batch_sizes = [i*32 for i in range(1, 11)]

for la, th, bs in product(lambidas, thetas, batch_sizes):

    # # -------------- MNIST ------------------- ##
    print("Lambda:", la, "Theta:", th, "Batch Size:", bs)
    test_mnist = MNIST(root='./data', train=False, download=True, transform=ToTensor())
    orig_loader = DataLoader(test_mnist,  batch_size = bs, shuffle=True)

    
    model = load_model('trained_models/CNN_mnist_downloaded.torch', Mnist_CNN_Classifier())
    # # Sanity check to verify performence on clean test data 

    # out = DDAL_test(orig_loader=orig_loader,drift_loader=None, model=model, size_batch = bs, theta = th, lambida = la)

    # with open(f'experiments_results/finetune_ddal/mnist_clean_test_la{la}_th{th}_bs{bs}.dict', 'wb') as f:
    #     pickle.dump(out, f)
    
    # print(out['Drift Detected'])
        
    ## Abrupt case withhold
        
    withhold_class = ImageFolder(root='data/transformed/mnist-w-0', transform=Compose([ToTensor(),Grayscale(num_output_channels=1)]))
    drift_loader = DataLoader(dataset=withhold_class, batch_size = bs)

    # out = DDAL_test(orig_loader=orig_loader,drift_loader=drift_loader, model=model, size_batch = bs, theta = th, lambida = la)

    # print(out['Drift Detected'])

    # with open(f'experiments_results/finetune_ddal/mnist_abrupt_w-0_la{la}_th{th}_bs{bs}.dict', 'wb') as f:
    #     pickle.dump(out, f)
        
    # Gradual case withhold

    out = DDAL_test_gradual(orig_loader=orig_loader,drift_loader=drift_loader, model=model, size_batch = bs, theta = th, lambida = la)

    with open(f'experiments_results/finetune_ddal/mnist_gradual_w-0_la{la}_th{th}_bs:{bs}.dict', 'wb') as f:
        pickle.dump(out, f)
        
    print(out['Drift Detected'])
    
    rotated = ImageFolder(root='data/transformed/mnist-rotated90', transform=Compose([ToTensor(),Grayscale(num_output_channels=1)]))
    drift_loader = DataLoader(dataset=rotated, batch_size = bs)
    
    out = DDAL_test_gradual(orig_loader=orig_loader,drift_loader=drift_loader, model=model, size_batch = bs, theta = th, lambida = la)

    with open(f'experiments_results/finetune_ddal/mnist_gradual_rotated_la{la}_th{th}_bs:{bs}.dict', 'wb') as f:
        pickle.dump(out, f)
        
    print(out['Drift Detected'])
    
    out = DDAL_test(orig_loader=orig_loader,drift_loader=drift_loader, model=model, size_batch = bs, theta = th, lambida = la)

    with open(f'experiments_results/finetune_ddal/mnist_abrupt_rotated_la{la}_th{th}_bs:{bs}.dict', 'wb') as f:
        pickle.dump(out, f)
        
    print(out['Drift Detected'])
    
    ## -------------- CIFAR ------------------- ##
    rotated = ImageFolder(root='data/transformed/cifar-rotated90', transform=ToTensor())
    drift_loader = DataLoader(dataset=rotated, batch_size = bs)
    test_cifar = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())
    orig_loader = DataLoader(test_cifar,  batch_size = bs, shuffle=True)

    model = load_model('trained_models/CNN_cifar_downloaded.torch', CIFAR_CNN_Classifier())
    
    # Sanity check to verify performence on clean test data 

    # out = DDAL_test(orig_loader=orig_loader,drift_loader=None, model=model, size_batch = bs, theta = th, lambida = la)
    
    # print(out['Drift Detected'])
    
    # with open(f'experiments_results/finetune_ddal/cifar_clean_test_la{la}_th{th}_bs{bs}.dict', 'wb') as f:
    #     pickle.dump(out, f)
        
    # Abrupy test withhold

    # out = DDAL_test(orig_loader=orig_loader,drift_loader=drift_loader, model=model)
    
    # print(out['Drift Detected'])

    # with open(f'experiments_results/finetune_ddal/cifar_rotate_abrupt_rotate_la{la}_th{th}_bs{bs}.dict', 'wb') as f:
    #     pickle.dump(out, f)

    out = DDAL_test_gradual(orig_loader=orig_loader,drift_loader=drift_loader, model=model)

    with open(f'experiments_results/finetune_ddal/cifar_rotate_gradual_rotate_la{la}_th{th}_bs{bs}.dict', 'wb') as f:
        pickle.dump(out, f)
        
    withhold_class = ImageFolder(root='data/transformed/cifar-w-0', transform=ToTensor())
    drift_loader = DataLoader(dataset=withhold_class, batch_size = bs)

    # out = DDAL_test(orig_loader=orig_loader,drift_loader=drift_loader, model=model)
    
    # print(out['Drift Detected'])

    # with open(f'experiments_results/finetune_ddal/cifar_abrupt_w-0_la{la}_th{th}_bs{bs}.dict', 'wb') as f:
    #     pickle.dump(out, f)

    out = DDAL_test_gradual(orig_loader=orig_loader,drift_loader=drift_loader, model=model)
    
    print(out['Drift Detected'])

    with open(f'experiments_results/finetune_ddal/cifar_gradual_w-0_la{la}_th{th}_bs{bs}.dict', 'wb') as f:
        pickle.dump(out, f)