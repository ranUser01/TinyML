from torchvision.transforms import Compose, ToTensor, Normalize, RandomRotation
from CNN_setup.datasets.datasets import CustomCIFAR10
import torchshow as ts
from torchvision.datasets import CIFAR10
import keyboard
import time
import matplotlib.pyplot as plt


transform_rotate = Compose([ToTensor(), 
                            RandomRotation (degrees = (90,90)), 
                            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


test_dataset = CIFAR10(root='./data', train=False, download=False, transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
rotated_data = CustomCIFAR10(root='../data', train=False, download=True, transform=transform_rotate)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

fig, axs = plt.subplots(1, 2)

for rotated_image, test_image in zip(rotated_data, test_dataset):
    rotated_image, rotated_label = rotated_image
    test_image, test_label = test_image

    axs[0].imshow(test_image.permute(1, 2, 0))
    axs[0].set_title("Test Dataset: " + classes[test_label])
    axs[0].axis('off')

    axs[1].imshow(rotated_image.permute(1, 2, 0))
    axs[1].set_title("Rotated Data: " + classes[rotated_label])
    axs[1].axis('off')

    plt.show()
    # time.sleep(1)