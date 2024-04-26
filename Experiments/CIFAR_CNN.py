import argparse
from torch.utils.data import DataLoader
from torch.nn.functional import relu, max_pool2d, dropout, softmax
import torch.nn as nn
from cnn_models_utils import train, save_model, evaluate
from torchvision.transforms import Compose, ToTensor, Normalize

# Cifar CNN
class CIFAR_CNN_Classifier(nn.Module):
    # That comes from cifar10_turorial from Pytorch official website
    # def __init__(self):
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(3, 6, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 10)

    # def forward(self, x):
    #     x = self.pool(relu(self.conv1(x)))
    #     x = self.pool(relu(self.conv2(x)))
    #     x = flatten(x, 1) # flatten all dimensions except batch
    #     x = relu(self.fc1(x))
    #     x = relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x
    
    #this comes from https://www.kaggle.com/code/sachinpatil1280/cifar-10-image-classification-cnn-89
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = max_pool2d
        self.dropout = dropout
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(relu(self.conv1(x)), kernel_size=2, stride=2)
        x = self.pool(relu(self.conv2(x)), kernel_size=2, stride=2)
        x = self.pool(relu(self.conv3(x)), kernel_size=2, stride=2)
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x, p=0.25)
        x = relu(self.fc1(x))
        x = self.dropout(x, p=0.5)
        x = self.fc2(x)
        return softmax(x)

    
def main(path_prefix:str = '../data/Mnist', local_data:bool = False, num_epochs=20):
    if local_data: # use locally stored Cifar data 
        raise Exception("Local data has not been implemented")
        from utils import ImageDFDataset

        # train
        train_dataset = ImageDFDataset(f"{path_prefix}/mnist_train.csv", label_col_name='label')
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # test 
        test_dataset = ImageDFDataset(f"{path_prefix}/mnist_train.csv",label_col_name='label')
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        

    else: # Download and use Pytorch's Cifar data
        from torchvision.datasets.cifar import CIFAR10 
        transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        batch_size = 32

        train_dataset = CIFAR10(root='../data', train=True, download=True, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = CIFAR10(root='../data', train=False, download=True, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model = train(dataloader = train_dataloader, model = CIFAR_CNN_Classifier(), num_epochs=num_epochs)
    if model is not None:
        save_model(path_dst="CNN_cifar_base.torch", model=model)
        
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    evaluate(test_dataloader, model, classes)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_prefix', type=str, default='../data/Cifar', help='Path prefix for data')
    parser.add_argument('-l', '--local_data', action='store_true', help='Use locally stored mnist data')
    args = parser.parse_args()
    
    main(path_prefix=args.path_prefix, local_data=args.local_data)