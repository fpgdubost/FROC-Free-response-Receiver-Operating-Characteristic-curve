import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def computeFROC(proba_map,ground_truth,nbr_of_thresholds):
    #INPUTS
    #proba_map : numpy array of dimension [number of image, xdim, ydim,...], values preferably in [0,1]
    #ground_truth: numpy array of dimension [number of image, xdim, ydim,...], values in {0,1}
    #nbr_of_thresholds: number of thresholds to compute to plot the FROC
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
            P = np.count_nonzero(ground_truth)
            TP = np.count_nonzero(thresholded_proba_map*ground_truth)
            FP = np.count_nonzero(thresholded_proba_map - (thresholded_proba_map*ground_truth))
            
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
    
def computeAndPlotFROC(proba_map,ground_truth,nbr_of_thresholds,save_path):
    #INPUTS
    #proba_map : numpy array of dimension [number of image, xdim, ydim,...], values preferably in [0,1]
    #ground_truth: numpy array of dimension [number of image, xdim, ydim,...], values in {0,1}
    #nbr_of_thresholds: number of thresholds to compute to plot the FROC
    #save_path: path to save the FROC plot
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
            P = np.count_nonzero(ground_truth)
            TP = np.count_nonzero(thresholded_proba_map*ground_truth)
            FP = np.count_nonzero(thresholded_proba_map - (thresholded_proba_map*ground_truth))
            
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
    
    #create artificial data
    ground_truth = np.random.randint(2, size=[nbr_img]+size_img)
    proba_map = np.random.randint(100, size=[nbr_img]+size_img)*1./100
    
    #compute sensitivity and FPavg
    sensitivity_list, FPavg_list = computeFROC(proba_map,ground_truth,nbr_of_thresholds)
    plotFROC(FPavg_list,sensitivity_list,save_path)
    
    #OR at once
#    computeAndPlotFROC(proba_map,ground_truth,nbr_of_thresholds,save_path)
    
    
    
    
    