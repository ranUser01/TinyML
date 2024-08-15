from incremental_ks.IncrementalKS.IKSSW import IKSSW
from CNN_setup.datasets.dataset_tools import GradualDrifttoader

def test_IKS_abrupt(orig_loader, drift_loader):
    res_dict = {}
    res_dict['Drift Detected'] = list()
    
    n = -1
    
    iks_list = [None for _ in range(28**2)]  # create a seperate IKS for each feature in the image 

    cur_loader = orig_loader
    for i in range(len(orig_loader)):
        iks_test_results = []
        ks_statistics = []

        if n == len(orig_loader):
            break
        else:
            n += 1

        ## Here the dataloader changes to simulate an abrupt drift 
        if n == len(orig_loader) // 2 and drift_loader is not None:
            cur_loader = drift_loader
            res_dict['Drift started at'] = n 

        for img, label in cur_loader: # it returns only 1 batch at a time
            # assert(len(label) == 1) # ensures that I take only 1 image
            img = img.reshape(1,-1).tolist()[0]

            for iks, feature_val in zip(iks_list, img):
                if n == 0 : ## if it does not setup reference windows 
                    for i in range(len(iks_list)):
                        iks_list[i] = IKSSW([feature_val])
                else: # set reference 
                    iks.Increment(feature_val)
                    ks_statistics.append(iks.KS())
                    test_res = iks.Test()
                    iks_test_results.append(test_res)
                    if test_res : 
                        print(f'Drift detected at: {n}')
                        res_dict['Drift Detected'].append(n)
                    res_dict[i] = (iks_test_results, ks_statistics)
            
            break
                    
    return res_dict
  

def test_IKS_gradual(orig_loader, drift_loader):
    res_dict = {}
    res_dict['Drift Detected'] = list()
    
    iks_list = [None for _ in range(28**2)]  # create a seperate IKS for each feature in the image 
    
    grad_loader = GradualDrifttoader(orig_loader,drift_loader, shift_step = 8)
    res_dict = {}
    res_dict['Drift Detected'] = list()
    res_dict['Drift started at'] = len(grad_loader)//2
        
    for i in range(len(grad_loader)):
        iks_test_results = []
        ks_statistics = []

        for batch in grad_loader:
            imgs, _ = batch
            for i in range(imgs.shape[0]): # it returns only 1 batch at a time
                img = imgs[i]
                # assert(len(label) == 1) # ensures that I take only 1 image
                img = img.reshape(1,-1).tolist()[0]

                for iks, feature_val in zip(iks_list, img):
                    if i == 0 : ## if it does not setup reference windows 
                        for i in range(len(iks_list)):
                            iks_list[i] = IKSSW([feature_val])
                    else : # set observation and remove old one 
                        iks.Increment(feature_val)
                        ks_statistics.append(iks.KS())
                        test_res = iks.Test()
                        iks_test_results.append(test_res)
                        if test_res : 
                            print(f'Drift detected at: {i}')
                            res_dict['Drift Detected'].append(i)
                        res_dict[i] = (iks_test_results, ks_statistics)
                        
                break
                        
    return res_dict