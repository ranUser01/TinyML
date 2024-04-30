from CNN_setup.utils.cnn_models_utils import load_model, get_probabilities, evaluate
from CNN_setup.run_CIFAR import CIFAR_CNN_Classifier
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, RandomRotation

transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_dataset = CIFAR10(root='../data', train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = load_model("models/CNN_cifar_base.torch",CIFAR_CNN_Classifier())
# probs = get_probabilities(test_dataloader,model)
# print(probs[0:1])

# Rotate images by 90 degrees 
transform = Compose([ToTensor(), RandomRotation (degrees = 90) ,Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test__rotatesdataset = CIFAR10(root='../data', train=False, download=True, transform=transform)


evaluate(test__rotatesdataset,model,classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'))