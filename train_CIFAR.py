import argparse
from torch.utils.data import DataLoader, random_split
from CNN_setup.utils.cnn_models_utils import train, save_model, evaluate, train_with_earlystop
from torchvision.transforms import Compose, ToTensor, Normalize
from CNN_setup.model.CIFAR_CNN import CIFAR_CNN_Classifier ##, CIFAR10_classes

def main(path_prefix:str = '../data/Mnist', local_data:bool = False, num_epochs=20):
    if local_data: # use locally stored Cifar data 
        raise Exception("Local data has not been implemented")
        from utils import ImageDFDataset

        # train
        train_dataset = ImageDFDataset(f"{path_prefix}/mnist_train.csv", label_col_name='label')
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # test 
        test_dataset = ImageDFDataset(f"{path_prefix}/mnist_train.csv",label_col_name='label')
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    else: # Download and use Pytorch's Cifar data
        from torchvision.datasets.cifar import CIFAR10 
        transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        batch_size = 32

        train_dataset = CIFAR10(root='../data', train=True, download=True, transform=transform)
        
        # split train to train and val
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        #val
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

        test_dataset = CIFAR10(root='../data', train=False, download=True, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    try:
        model = train_with_earlystop(dataloader = train_dataloader, model = CIFAR_CNN_Classifier(),
                                            lr= 0.0009, num_epochs = 30,
                                            patience=5, dataloader_val = val_dataloader)
        if model is not None:
            save_model(path_dst="CNN_cifar_downloaded.torch", model=model)
            
    except Exception as e:
        print('Model saving unsuccessful')
        raise(e)
    
    CIFAR10_classes = tuple([_ for _ in range(0, 10, 1)]) 
    
    evaluate(test_dataloader, model, CIFAR10_classes)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_prefix', type=str, default='../data/Cifar', help='Path prefix for data')
    parser.add_argument('-l', '--local_data', action='store_true', help='Use locally stored mnist data')
    args = parser.parse_args()
    
    main(path_prefix=args.path_prefix, local_data=args.local_data)