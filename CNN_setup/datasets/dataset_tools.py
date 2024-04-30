from torchvision.datasets import ImageFolder
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