from tqdm import tqdm
from torch import save, no_grad, argmax, load
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn

from torchvision.transforms import Compose
from torchvision.transforms.functional import rotate

from pathlib import Path
from typing import Union

def train(dataloader:DataLoader,num_epochs:int = 10, lr:float=0.001, model: nn.Module = None):
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr)

        pbar = tqdm(total=num_epochs, desc="Training progress", ncols=100)

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
            print(f"")

        pbar.close()
        return model

    except Exception as e:
        raise Exception(f"An error occurred during training: {e}")
    
def train_with_earlystop(dataloader:DataLoader, dataloader_val:DataLoader = None, num_epochs:int = 10, 
                         lr:float=0.001, model: nn.Module = None, patience:int = 3):
    if dataloader_val is None:
        print("Dataloader_val has not been provided: proceeding with training without it")
        return train(dataloader=dataloader,num_epochs=num_epochs,lr=lr,model=model)
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr)

        pbar = tqdm(total=num_epochs, desc="Training progress", ncols=150)

        best_loss = float('inf')
        early_stop_counter = 0

        for epoch in range(num_epochs):
            # print(f'epoch: {epoch}\n')
            epoch_loss = 0
            for inputs, labels in dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)

            # Validation
            val_loss = 0
            with no_grad():
                for val_inputs, val_labels in dataloader_val:
                    val_outputs = model(val_inputs)
                    val_loss += criterion(val_outputs, val_labels).item()
            
            avg_val_loss = val_loss / len(dataloader_val)
            pbar.set_postfix({'Epoch Loss': avg_loss, 'Validation Loss': avg_val_loss
                              , 'Best val loss': best_loss
                              , 'Patience Counter': early_stop_counter})
            pbar.update()

            # Check for early stopping
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print("Early stopping triggered!")
                    break

        pbar.close()
        return model

    except Exception as e:
        raise Exception(f"An error occurred during training: {e}")

def evaluate(test_dataloader:DataLoader, model:nn.Module, classes:tuple = tuple([_ for _ in range(0, 10, 1)])):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = model(images)
            predictions = argmax(outputs, 1) 
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        if total_pred[classname] != 0:
            accuracy = 100 * float(correct_count) / total_pred[classname]
        else:
            accuracy = 0.0
        print(f'Accuracy for class: {classname} is {accuracy:.1f} %')
    
def evaluate_batch(batch, model:nn.Module, classes:tuple = tuple([_ for _ in range(0, 10, 1)]) , save_output_layer:bool = False):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with no_grad():
        images, labels = batch
        outputs = model(images)
        predictions = argmax(outputs, 1) 
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

    accuracy_dict = {}
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] != 0:
            accuracy = 100 * float(correct_count) / total_pred[classname]
        else:
            accuracy = 0.0
        accuracy_dict[classname] = accuracy

    total_accuracy = sum(correct_pred.values()) / sum(total_pred.values())
    accuracy_dict['Total Accuracy'] = total_accuracy
    
    # if save_output_layer:
    #     import time, pickle
    #     with open('experiments_results/model_output.dict', 'wb') as f:
    #         pickle.dump(out, f)

    return accuracy_dict
    
def get_probabilities(test_dataloader:DataLoader, model:nn.Module):
    probabilities = []

    with no_grad():
        for data in test_dataloader:
            images, _ = data
            outputs = model(images)
            probabilities.append(outputs)

    return probabilities

def get_probabilities_batch(batch, model:nn.Module):

    with no_grad():
        images, _ = batch
        outputs = model(images)
        
    return outputs
    
def save_model(path_dst: Union[str, Path] = "CNN_mnist_base.torch", model = None, folder: str = 'trained_models'):
    try:
        checkpoint = {'state_dict': model.state_dict()}
        save(checkpoint, f"{folder}/{path_dst}")
        # save(model, f"models/{path_dst}")
    except Exception as e:
        print('Model saving unsuccessful')
        raise(e)
    
def load_model(path_dst:str = "CNN_mnist_base.torch", model:nn.Module = None):
    wab_dict = load(path_dst)
    model.load_state_dict(state_dict=wab_dict['state_dict'])
    return model

def simulate_drift_dataloader():
    drift_transform = Compose(rotate(angle = 90))
    return drift_transform