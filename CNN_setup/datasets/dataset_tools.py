from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision.transforms import ToTensor, Compose
import os
from torch import cat

def split_data_by_class(data:Dataset, class_index):
    """
    Splits the given data into two lists based on the class index.

    Args:
        data (list): The input data to be split.
        class_index (int): The index of the class to split the data by.

    Returns:
        tuple: A tuple containing two lists. The first list contains the data samples
               that belong to the specified class, and the second list contains the
               data samples that do not belong to the specified class.
    """
    w_class = []
    wo_class = []
    for i, l in data:
        if l == class_index:
            w_class.append((i,l))
        else:
            wo_class.append((i,l))

    return w_class, wo_class

def save_dataset(data:Dataset, path:str = 'data/transformed/cifar-div-by-class', transform = None, args = ()):
    """
    Save the dataset images to the specified path, optionally applying a transformation.

    Args:
        data: The dataset to save, in the form of a list of tuples (image, label).
        path (str): The path to save the images. Default is 'data/transformed/cifar-div-by-class'.
        transform: The transformation function to apply to each image. Default is None.
        args: Additional arguments to pass to the transformation function. Default is an empty tuple.
    """
    for i, (img, label) in enumerate(data):
        # Apply the transformation if one was provided
        if transform is not None:
            img = transform(img,  *args)
        os.makedirs(f'{path}/{label}', exist_ok=True)
        img.save(f'{path}/{label}/img_{i}.png',)
    pass

def combine_datasets(dataset1:Dataset, dataset2:Dataset, proportion1:float):
    assert 1 >= proportion1 > 0 
    proportion2 = 1 - proportion1
    total_samples = int((len(dataset1) * proportion1) + (len(dataset2) * proportion2))

    threshold1 = int(total_samples * proportion1)
    threshold2 = int(total_samples * proportion2)

    subset1 = Subset(dataset1, range(threshold1))
    subset2 = Subset(dataset2, range(threshold2))
    
    combined_data = ConcatDataset([subset1, subset2])
    
    return combined_data

class Flatten:
    '''
    Flattens a tensor to one dimension
    '''
    def __call__(self, tensor):
        return tensor.view(-1)

    def __repr__(self):
        return self.__class__.__name__ + '()'

def load_dataset(path:str, flat:bool = False):
    if flat:
        return ImageFolder(root = f'{path}', transform=Compose( [ToTensor(), Flatten()]  ))
    else:
        return ImageFolder(root = f'{path}', transform=ToTensor())
    
    
class GradualDrifttoader:
    def __init__(self, orig_loader, drift_loader, shift_step = 8):
        self.loader1 = iter(orig_loader)
        self.loader2 = iter(drift_loader)
        self.shift_step = shift_step
        self.current_shift = 0
        self.steps_so_far = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.steps_so_far += 1

        batch1 = next(self.loader1, None)
        batch2 = next(self.loader2, None)

        if batch1 is None or batch2 is None:
            raise StopIteration

        if self.current_shift == 0:
            combined_batch = batch1
        elif self.current_shift >= 32:
            combined_batch = batch2
        else:
            # Take the first (batch_size - current_shift) samples from batch1
            batch1 = batch1[:32 - self.current_shift]

            # Take the first current_shift samples from batch2
            batch2 = batch2[:self.current_shift]
        
            # Take the first (batch_size - current_shift) samples from batch1
            inputs1, labels1 = batch1[0][:32 - self.current_shift], batch1[1][:32 - self.current_shift]

            # Take the first current_shift samples from batch2
            inputs2, labels2 = batch2[0][:self.current_shift], batch2[1][:self.current_shift]

            # Combine the inputs and labels from the two batches
            combined_inputs = cat([inputs1, inputs2])
            combined_labels = cat([labels1, labels2])

            # Combine the inputs and labels into a single batch
            combined_batch = (combined_inputs, combined_labels)

        # Increase the shift for the next batch
        if self.steps_so_far > self.__len__()//2:
            self.current_shift += self.shift_step

        return combined_batch
    
    def __len__(self):
        return len(self.loader1) + self.shift_step - 1
    
    
    
