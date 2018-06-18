import numpy as np
from froc import computeFROC, plotFROC

if __name__ == '__main__':
    
    np.random.seed(1)    
    
    #parameters
    nbr_img = 20
    size_img = [10,10]
    save_path = 'FROC_example1.pdf'
    nbr_of_thresholds = 40
    range_threshold = [.1,.5]
    allowedDistance = 2
    
    #create artificial data
    ground_truth = np.random.randint(2, size=[nbr_img]+size_img)
    proba_map = np.random.randint(100, size=[nbr_img]+size_img)*1./100
    
    #compute FROC    
    sensitivity_list, FPavg_list, _ = computeFROC(proba_map,ground_truth, allowedDistance, nbr_of_thresholds, range_threshold)
    print 'computed FROC'
    
    #plot FROC
    plotFROC(FPavg_list,sensitivity_list,save_path)
    print 'plotted FROC'
    
    
    
    