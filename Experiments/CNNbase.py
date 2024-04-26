import pandas as pd
from numpy import float32
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn.functional import relu

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

# Custom dataset for image data frames, from any path given. label_col_name must be given 
class ImageDFDataset(Dataset):
    def __init__(self, path:str, label_col_name: str):
        df = pd.read_csv(path)
        self.images = df.drop(columns= label_col_name).to_numpy(float32)
        self.labels = df[label_col_name].to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx:int):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label
    

def unpickle(file:str):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def cifarToDf (path:str = "../data/cifar-10-python\data_batch_1"):
    cifar = unpickle(path)
    cifar_df = pd.DataFrame(cifar[b'data'])
    cifar_df['label'] = cifar[b'labels']

    return cifar_df

def train(path:str, local_data:bool = False):
    if local_data: # use locally stored mnist data 
        data = pd.read_csv("../data/Mnist/mnist_train.csv")
        dataset = ImageDFDataset(data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    else: # Download and use Pytorch's Mnist data
        from torchvision.datasets import mnist
        dataloader = mnist(root='./data', train=True,
                    download=True)

    # Create an instance of your CNN model
    model = Mnist_CNN_Classifier()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs, labels = batch

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss for monitoring the training progress
            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
def main():


if __name__ == '__main__':
    main()
    print("CNN has been trained")