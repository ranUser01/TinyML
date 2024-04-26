from tqdm import tqdm
from torch import save, no_grad, max
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn

def train(dataloader:DataLoader,num_epochs:int = 10, lr:float=0.001, model: nn.Module = None):
    try:
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
    save(model, f"models/{path_dst}")

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