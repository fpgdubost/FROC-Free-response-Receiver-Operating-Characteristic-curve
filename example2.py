import numpy as np
from froc import computeAndPlotFROC
import imageio

if __name__ == '__main__':
    
    np.random.seed(1)    
    
    #parameters
    save_path = 'FROC_example2.pdf'
    nbr_of_thresholds = 40
    range_threshold = [0.1,.99]
    allowedDistance = 10
    write_thresholds = True
    
    #read the data
    ground_truth = np.expand_dims(imageio.imread('ground_truth.png')[:,:,0],axis=0)
    proba_map = np.expand_dims(imageio.imread('proba_map.png')[:,:,0],axis=0)
    print ground_truth.shape
    print proba_map.shape  
    
    #plot FROC
    computeAndPlotFROC(proba_map,ground_truth, allowedDistance, nbr_of_thresholds, range_threshold, save_path, write_thresholds)
  