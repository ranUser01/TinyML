import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import relu
from torch import save, no_grad, max

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


from tqdm import tqdm

def train(dataloader:DataLoader,num_epochs:int = 10, lr:float=0.001):
    try:
        model = Mnist_CNN_Classifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr)

        pbar = tqdm(total=num_epochs, desc="Training progress")

        for epoch in range(num_epochs):
            epoch_loss = 0
            for inputs, labels in dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            pbar.set_postfix({'Epoch Loss': avg_loss})
            pbar.update()

            print(f"Epoch: {epoch+1}, Average Loss: {avg_loss}")

        pbar.close()
        return model

    except Exception as e:
        raise Exception(f"An error occurred during training: {e}")

    
def save_model(path_dst:str = "CNN_mnist_base.torch", model = None):
    save(model, path_dst)


def evaluate(test_dataloader:DataLoader, model:nn.Module):
    classes = tuple([_ for _ in range(0, 10, 1)])
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = model(images)
            _, predictions = max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname} is {accuracy:.1f} %')

    total_accuracy = sum(correct_pred.values()) / sum(total_pred.values())
    print(f'Total Accuracy: {total_accuracy * 100:.1f} %')

    
def main(path_prefix:str = '../data/Mnist', local_data:bool = True):
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
        model = train(train_dataloader)
        if model is not None:
            save_model(model=model)
    except Exception as e:
        print('Model saving unsuccessful')
        raise(e)
    
    evaluate(test_dataloader, model)
    
if __name__ == '__main__':
    main()