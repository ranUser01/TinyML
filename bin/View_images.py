from CNN_setup.utils.cnn_models_utils import load_model
from CNN_setup.run_CIFAR import CIFAR_CNN_Classifier
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize, RandomRotation
import matplotlib.pyplot as plt

import torchshow as ts


transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_dataset = CIFAR10(root='../data', train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = load_model("models/CNN_cifar_downloaded.torch",CIFAR_CNN_Classifier())

# Rotate images by 90 degrees 
transform = Compose([ToTensor(), RandomRotation (degrees = 90) ,Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test__rotated_dataset = CIFAR10(root='../data', train=False, download=True, transform=transform)
test_Rrotated_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

for  image, label in test_dataset:
    print(classes[label])
    ts.show(image)
