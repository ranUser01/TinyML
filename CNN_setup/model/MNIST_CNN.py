from torch.nn.functional import relu, dropout, max_pool2d, softmax
import torch.nn as nn

# Basic CNN 
class Mnist_Linerar_NN_Classifier(nn.Module):
    def __init__(self):
        super(Mnist_Linerar_NN_Classifier, self).__init__()
        self.linear1 = nn.Linear(784,250)
        self.linear2 = nn.Linear(250,100)
        self.linear3 = nn.Linear(100,10)

    def forward(self, x):
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
class Mnist_CNN_Classifier(nn.Module):
    def __init__(self):
        super(Mnist_CNN_Classifier, self).__init__()
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
        return softmax(input=x, dim=1)

class Mnist_CNN_Classifier_wo_0(nn.Module):
    def __init__(self):
        super(Mnist_CNN_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 9)

    def forward(self, x):
        x = relu(max_pool2d(self.conv1(x), 2))
        x = relu(max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = relu(self.fc1(x))
        x = dropout(x, training=self.training)
        x = self.fc2(x)
        return softmax(input=x, dim=1)
