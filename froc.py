import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import euclidean

def computeConfMatElements(thresholded_proba_map, ground_truth, allowedDistance):
    
    if allowedDistance == 0 and type(ground_truth) == np.ndarray:
        P = np.count_nonzero(ground_truth)
        TP = np.count_nonzero(thresholded_proba_map*ground_truth)
        FP = np.count_nonzero(thresholded_proba_map - (thresholded_proba_map*ground_truth))    
    else:
    
        #reformat ground truth to a list  
        if type(ground_truth) == np.ndarray:
            #convert ground truth binary map to list of coordinates
            labels, num_features = ndimage.label(ground_truth)
            list_gt = ndimage.measurements.center_of_mass(ground_truth, labels, range(1,num_features+1))   
        elif type(ground_truth) == list:        
            list_gt = ground_truth        
        else:
            raise ValueError('ground_truth should be either of type list or ndarray and is of type ' + str(type(ground_truth)))
        
        #reformat thresholded_proba_map to a list
        labels, num_features = ndimage.label(thresholded_proba_map)
        list_proba_map = ndimage.measurements.center_of_mass(thresholded_proba_map, labels, range(1,num_features+1)) 
         
        #compute P, TP and FP  
        FP = 0
        TP = 0
        P = len(list_gt)
        #iterate over the lists
        for point_pm in list_proba_map:
            for point_gt in list_gt:                           
                if euclidean(point_pm,point_gt) < allowedDistance:
                    TP += 1
                else:
                    FP += 1
                                 
    return P,TP,FP
        
def computeFROC(proba_map,ground_truth,nbr_of_thresholds, allowedDistance):
    #INPUTS
    #proba_map : numpy array of dimension [number of image, xdim, ydim,...], values preferably in [0,1]
    #ground_truth: numpy array of dimension [number of image, xdim, ydim,...], values in {0,1}; or list of coordinates
    #nbr_of_thresholds: number of thresholds to compute to plot the FROC
    #allowedDistance: euclidian distance distance in pixels to consider a detection as valid (anisotropy not considered in the implementation)    
    #OUTPUTS
    #sensitivity_list_treshold: list of average sensitivy over the set of images for increasing thresholds
    #FPavg_list_treshold: list of average FP over the set of images for increasing thresholds
    
    #define the thresholds
    threshold_list = (np.linspace(np.min(proba_map),np.max(proba_map),nbr_of_thresholds)).tolist()
    
    sensitivity_list_treshold = []
    FPavg_list_treshold = []
    #loop ovedr thresholds
    for threshold in threshold_list:
        sensitivity_list_proba_map = []
        FP_list_proba_map = []
        #loop over proba map
        for i in range(len(proba_map)):
            #threshold the proba map
            thresholded_proba_map = np.zeros(np.shape(proba_map[i]))
            thresholded_proba_map[proba_map[i] >= threshold] = 1
                   
            #compute P, TP, and FP for this threshold and this proba map
            P,TP,FP = computeConfMatElements(thresholded_proba_map, ground_truth[i], allowedDistance)       
            
            #append results to list
            sensitivity_list_proba_map.append(TP*1./P)
            FP_list_proba_map.append(FP)
        
        #average sensitivity and FP over the proba map, for a given threshold
        sensitivity_list_treshold.append(np.mean(sensitivity_list_proba_map))
        FPavg_list_treshold.append(np.mean(FP_list_proba_map))    
        
    return sensitivity_list_treshold, FPavg_list_treshold

def plotFROC(x,y,save_path):
    plt.figure()
    plt.plot(x,y, 'o-') 
    plt.xlabel('FPavg')
    plt.ylabel('Sensitivity')
    plt.savefig(save_path)
    #plt.show()
    
def computeAndPlotFROC(proba_map,ground_truth,nbr_of_thresholds, allowedDistance, save_path):
    #INPUTS
    #proba_map : numpy array of dimension [number of image, xdim, ydim,...], values preferably in [0,1]
    #ground_truth: numpy array of dimension [number of image, xdim, ydim,...], values in {0,1}; or list of coordinates
    #nbr_of_thresholds: number of thresholds to compute to plot the FROC
    #allowedDistance: euclidian distance distance in pixels to consider a detection as valid (anisotropy not considered in the implementation)
    #save_path: path to save the FROC plot
    #allowedDistance: integer, number maximum of pixels between a ground truth dot and a true positive dot
    #typeGT: 'img' or 'list'
    #OUTPUTS
    #sensitivity_list_treshold: list of average sensitivy over the set of images for increasing thresholds
    #FPavg_list_treshold: list of average FP over the set of images for increasing thresholds
    
    
    #define the thresholds
    threshold_list = (np.linspace(np.min(proba_map),np.max(proba_map),nbr_of_thresholds)).tolist()
    
    sensitivity_list_treshold = []
    FPavg_list_treshold = []
    #loop ovedr thresholds
    for threshold in threshold_list:
        sensitivity_list_proba_map = []
        FP_list_proba_map = []
        #loop over proba map
        for i in range(len(proba_map)):
            #threshold the proba map
            thresholded_proba_map = np.zeros(np.shape(proba_map[i]))
            thresholded_proba_map[proba_map[i] >= threshold] = 1
                   
            #compute P, TP, and FP for this threshold and this proba map
            P,TP,FP = computeConfMatElements(thresholded_proba_map, ground_truth, allowedDistance) 
            
            #append results to list
            sensitivity_list_proba_map.append(TP*1./P)
            FP_list_proba_map.append(FP)
        
        #average sensitivity and FP over the proba map, for a given threshold
        sensitivity_list_treshold.append(np.mean(sensitivity_list_proba_map))
        FPavg_list_treshold.append(np.mean(FP_list_proba_map)) 
        
    plt.figure()
    plt.plot(FPavg_list_treshold,sensitivity_list_treshold, 'o-') 
    plt.xlabel('FPavg')
    plt.ylabel('Sensitivity')
    plt.savefig(save_path)


if __name__ == '__main__':
    
    #example
    nbr_img = 20
    size_img = [10,10]
    save_path = 'FROC.pdf'
    nbr_of_thresholds = 40
    allowedDistance = 5
    typeGT = 'list'
    
    #create artificial data
    ground_truth = np.random.randint(2, size=[nbr_img]+size_img)
    proba_map = np.random.randint(100, size=[nbr_img]+size_img)*1./100
    
    #compute sensitivity and FPavg
    sensitivity_list, FPavg_list = computeFROC(proba_map,ground_truth,nbr_of_thresholds, allowedDistance)
    plotFROC(FPavg_list,sensitivity_list,save_path)
    
    #OR at once
#    computeAndPlotFROC(proba_map,ground_truth,nbr_of_thresholds,save_path)
    
    
    
    
    