from torch.nn.functional import relu, max_pool2d, dropout, softmax
import torch.nn as nn

# Cifar CNN
class CIFAR_CNN_Classifier(nn.Module):
    #This comes from https://www.kaggle.com/code/sachinpatil1280/cifar-10-image-classification-cnn-89
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
        return softmax(input=x, dim=1); 
    
class CIFAR_CNN_Classifier_Simple(nn.Module):
    #This comes from cifar10_turorial from Pytorch official website
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(relu(self.conv1(x)))
        x = self.pool(relu(self.conv2(x)))
        x = flatten(x, 1) # flatten all dimensions except batch
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x
