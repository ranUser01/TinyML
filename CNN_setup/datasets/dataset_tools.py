from torchvision.datasets import ImageFolder
from PIL import Image
import os

def split_data_by_class(data, class_index):
    w_class = []
    wo_class = []
    for i, l in data:
        if l == class_index:
            w_class.append((i,l))
        else:
            wo_class.append((i,l))

    return w_class, wo_class

def save_dataset(data, path:str = 'data/transformed/cifar-div-by-class', transform = None, args = ()):
    for i, (img, label) in enumerate(data):
        # Apply the transformation if one was provided
        if transform is not None:
            img = transform(img,  *args)
        os.makedirs(f'{path}/{label}', exist_ok=True)
        img.save(f'{path}/{label}/img_{i}.png')