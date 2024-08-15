from torch import max
from DDAL.ddal import DDAL_detector
from CNN_setup.utils.cnn_models_utils import evaluate_batch, get_probabilities_batch
from CNN_setup.datasets.dataset_tools import GradualDrifttoader
import warnings

def DDAL_test(orig_loader, drift_loader, model, size_batch = 32, theta = 0.005, lambida = 0.95):
    
    """
    This function runs and tests performance of DDAL within an abrupt setting.
    
    if drift_loader is none then only orig_loader is used 
    
    """
    
    if drift_loader is None:
        warnings.warn("drift_loader is none so only orig_loader is used ", UserWarning)
    
    res_dict = {}
    res_dict['Drift Detected'] = list()
    CDD_DDAL = DDAL_detector(size_batch = size_batch,theta = theta, lambida = lambida)
    n = -1
    cur_loader = orig_loader
    for i in range(len(orig_loader)):
        if n == len(orig_loader):
            break
        else:
            n += 1
        
        ## Here the dataloader changes to simulate an abrupt drift 
        if n == len(orig_loader) // 2 and drift_loader is not None:
            cur_loader = drift_loader
            res_dict['Drift started at'] = n 
        
        for b in cur_loader:
            res_dict[n] = evaluate_batch(batch=b,model=model)
            probs = get_probabilities_batch(batch=b,model=model)
            max_values, _ = max(probs, dim=1, keepdim=True)
            CDD_DDAL.count_tensor(max_values)
            break
        
        CDD_DDAL.compute_current_density()
        if CDD_DDAL.detection_module():
            res_dict['Drift Detected'].append(n)
            CDD_DDAL.reset()
            
    return res_dict

def DDAL_test_gradual(orig_loader, drift_loader, model, size_batch = 32, theta = 0.005, lambida = 0.95):
    
    """
    This function runs and tests performance of DDAL within an gradual setting.
    
    if drift_loader is none then only orig_loader is used 
    
    """
    
    grad_loader = GradualDrifttoader(orig_loader,drift_loader)
    res_dict = {}
    res_dict['Drift Detected'] = list()
    res_dict['Drift started at'] = len(grad_loader)//2
    CDD_DDAL = DDAL_detector(size_batch = size_batch,theta = theta, lambida = lambida)
    for i in range(len(grad_loader)):
        for b in grad_loader:
            res_dict[i] = evaluate_batch(batch=b,model=model)
            probs = get_probabilities_batch(batch=b,model=model)
            max_values, _ = max(probs, dim=1, keepdim=True)
            CDD_DDAL.count_tensor(max_values)
            break
        
        CDD_DDAL.compute_current_density()
        if CDD_DDAL.detection_module():
            res_dict['Drift Detected'].append(i)
            CDD_DDAL.reset()
            
    return res_dict