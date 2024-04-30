from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset, Subset
import os

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