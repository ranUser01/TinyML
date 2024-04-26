import argparse
from torch.utils.data import DataLoader
from torch.nn.functional import relu, dropout, max_pool2d, log_softmax
import torch.nn as nn
from cnn_models_utils import train, save_model, evaluate
from torchvision.transforms import ToTensor

# Basic CNN 
class Mnist_CNN_Classifier(nn.Module):
    def __init__(self):
        super(Mnist_CNN_Classifier, self).__init__()
        self.linear1 = nn.Linear(784,250)
        self.linear2 = nn.Linear(250,100)
        self.linear3 = nn.Linear(100,10)

    def forward(self, x):
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = relu(max_pool2d(self.conv1(x), 2))
        x = relu(max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = relu(self.fc1(x))
        x = dropout(x, training=self.training)
        x = self.fc2(x)
        log_softmax(x)
        return x
    
def main(path_prefix:str = '../data/Mnist', local_data:bool = True):
    if local_data: # use locally stored mnist data 
        from utils import ImageDFDataset

        # train
        train_dataset = ImageDFDataset(f"{path_prefix}/mnist_train.csv", label_col_name='label')
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # test 
        test_dataset = ImageDFDataset(f"{path_prefix}/mnist_train.csv",label_col_name='label')
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
        try:
            model = train(dataloader = train_dataloader, model = Mnist_CNN_Classifier())
            if model is not None:
                save_model(path_dst="CNN_mnist_local.torch", model=model)
        except Exception as e:
            print('Model saving unsuccessful')
            raise(e)
    
        
    else: # Download and use Pytorch's Mnist data
        from torchvision.datasets.mnist import MNIST 
        train_dataset = MNIST(root='./data', train=True, download=True, transform=ToTensor())
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        test_dataset = MNIST(root='./data', train=False, download=False, transform=ToTensor())
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        
        try:
            model = train(dataloader = train_dataloader, model = Net())
            if model is not None:
                save_model(path_dst="CNN_mnist_downloaded.torch", model=model)
        except Exception as e:
            print('Model saving unsuccessful')
            raise(e)
    
    evaluate(test_dataloader, model)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_prefix', type=str, default='../data/Mnist', help='Path prefix for data')
    parser.add_argument('-l', '--local_data', action='store_true', help='Use locally stored mnist data')
    args = parser.parse_args()
    
    main(path_prefix=args.path_prefix, local_data=args.local_data)