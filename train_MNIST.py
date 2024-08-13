import argparse
from torch.utils.data import DataLoader, random_split
from CNN_setup.utils.cnn_models_utils import train_with_earlystop, save_model, evaluate 
from torchvision.transforms import ToTensor
from CNN_setup.model.MNIST_CNN import Mnist_CNN_Classifier, Mnist_Linerar_NN_Classifier

def main(path_prefix:str = '../data/Mnist', local_data:bool = True, withhold_data:bool = True):
    if local_data and not withhold_data: # use locally stored mnist data 
        print("Using downloaded data")
        raise(Exception('Local run is unfinished'))
        from CNN_setup.utils.utils import ImageDFDataset

        # train
        train_dataset = ImageDFDataset(f"{path_prefix}/mnist_train.csv", label_col_name='label')
        val_dataset = ImageDFDataset(f"{path_prefix}/mnist_train.csv", label_col_name='label')
        
        # split train to train and val
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        #val
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

        # test 
        test_dataset = ImageDFDataset(f"{path_prefix}/mnist_train.csv",label_col_name='label')
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
        try:
            model = train_with_earlystop(dataloader = train_dataloader, model = Mnist_Linerar_NN_Classifier(),
                                        lr= 0.0009, num_epochs = 30,
                                        patience=3, dataloader_val = val_dataloader)
            if model is not None:
                save_model(path_dst="CNN_mnist_local.torch", model=model)
        except Exception as e:
            print('Model saving unsuccessful')
            raise(e)
        
    elif withhold_data and not local_data: # use locally stored mnist data without 0 class
        print("Using withhold data")
        from torchvision.transforms import ToTensor, Compose, Grayscale
        from torchvision.datasets import ImageFolder
        
        orig_data_wo_0 = ImageFolder(root='data/transformed/train_mnist/mnist-wo-0',
                                      transform=Compose([ToTensor(),Grayscale(num_output_channels=1)]))
    
        # split train to train and val
        train_size = int(0.9 * len(orig_data_wo_0))
        val_size = len(orig_data_wo_0) - train_size
        
        train_dataset, val_dataset = random_split(orig_data_wo_0, [train_size, val_size])
        
        #val
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

        # test 
        test_dataset = ImageFolder(root='data/transformed/test_mnist/mnist-wo-0',
                                      transform=Compose([ToTensor(),Grayscale(num_output_channels=1)]))
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
        try:
            model = train_with_earlystop(dataloader = train_dataloader, model = Mnist_CNN_Classifier(),
                                        lr= 0.0009, num_epochs = 30,
                                        patience=3, dataloader_val = val_dataloader)
            if model is not None:
                save_model(path_dst="CNN_mnist_wo_0.torch", model=model)
        except Exception as e:
            print('Model saving unsuccessful')
            raise(e)
    
        
    else: # Download and use Pytorch's Mnist data
        print("Using Pytorch's mnist data")
        from torchvision.datasets.mnist import MNIST 
        from torchvision.transforms import ToTensor
        train_dataset = MNIST(root='./data', train=True, download=True, transform=ToTensor())
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # split train to train and val
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        #val
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        
        test_dataset = MNIST(root='./data', train=False, download=False, transform=ToTensor())
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        
        try:
            model = train_with_earlystop(dataloader = train_dataloader, model = Mnist_CNN_Classifier(),
                                        lr= 0.0009, num_epochs = 30,
                                        patience=3, dataloader_val = val_dataloader)
            if model is not None:
                save_model(path_dst="CNN_mnist_downloaded.torch", model=model)
        except Exception as e:
            print('Model saving unsuccessful')
            raise(e)
        
    if withhold_data:
        classes = tuple([_ for _ in range(0, 9, 1)])
    else: 
        classes = tuple([_ for _ in range(0, 10, 1)])
        
    evaluate(test_dataloader, model, classes=classes)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_prefix', type=str, default='../data/Mnist', help='Path prefix for data')
    parser.add_argument('-l', '--local_data', action='store_true', help='Use locally stored mnist data')
    parser.add_argument('-w', '--withhold_data', action='store_true', help='Use withhold mnist data')
    args = parser.parse_args()
    
    main(path_prefix=args.path_prefix, local_data=args.local_data)