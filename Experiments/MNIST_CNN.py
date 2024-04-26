import argparse
from torch.utils.data import DataLoader
from torch.nn.functional import relu
import torch.nn as nn
from cnn_models_utils import train, save_model, evaluate

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
    
def main(path_prefix:str = '../data/Mnist', local_data:bool = False):
    if local_data: # use locally stored mnist data 
        from utils import ImageDFDataset

        # train
        train_dataset = ImageDFDataset(f"{path_prefix}/mnist_train.csv", label_col_name='label')
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # test 
        test_dataset = ImageDFDataset(f"{path_prefix}/mnist_train.csv",label_col_name='label')
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    else: # Download and use Pytorch's Mnist data
        from torchvision.datasets.mnist import MNIST 
        train_dataloader = MNIST(root='./data', train=True, download=True)
        test_dataloader = MNIST(root='./data', train=False, download=False)

    try:
        model = train(dataloader = train_dataloader, model = Mnist_CNN_Classifier())
        if model is not None:
            save_model(model=model)
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