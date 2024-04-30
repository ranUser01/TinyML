from pandas import DataFrame
import pickle


def unpickle(file:str):
    '''
    file: path to the Cifar batch file

    returns dict: 
    dict_keys([b'batch_label', b'labels', b'data', b'filenames'])

    Used to open pickled Cifar10 files 
    Coming from "https://www.cs.toronto.edu/~kriz/cifar.html"

    '''
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def cifarToDf (path:str = "../data/cifar-10-python\data_batch_1"):
    '''
    file: path to the Cifar batch file

    Loads and converts a Cifar file to a Pandas DataFrame 
    '''
    cifar = unpickle(path)
    cifar_df = DataFrame(cifar[b'data'])
    cifar_df['label'] = cifar[b'labels']

    return cifar_df