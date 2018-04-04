import numpy as np
from froc import computeAndPlotFROC

if __name__ == '__main__':
    
    np.random.seed(1)    
    
    #parameters
    nbr_img = 20
    size_img = [10,10]
    save_path = 'FROC_example1.pdf'
    nbr_of_thresholds = 40
    range_threshold = [.1,.5]
    allowedDistance = 2
    write_thresholds = False
    
    #create artificial data
    ground_truth = np.random.randint(2, size=[nbr_img]+size_img)
    proba_map = np.random.randint(100, size=[nbr_img]+size_img)*1./100
    
    #plot FROC
    computeAndPlotFROC(proba_map,ground_truth, allowedDistance, nbr_of_thresholds, range_threshold, save_path, write_thresholds)
    
    
    
    
    