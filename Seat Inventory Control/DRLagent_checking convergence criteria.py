# This script is used to generate test plots of optimalRewardPercentage and loadFactor of the best model (best among all the models saved at the checkpoints)

# TODO: may need to change filepath, testData_filename, epsMaxRPind, MATLABepisodeIndex
# TODO: For the case of 3 fare classes, uncomment lines with fares_fc3 

import pickle

import numpy as np
import matplotlib.pyplot as plt


# from armClasses_dynamicPricing import *
from  DRLagent_armClasses_3fClasses_Tb1b2b3_SIC import infoLogger,running_mean



bumpingCostFactor = 2
cancelProbsSetting_array = [1,2,3]
meanNarrivalsSetting_array = [1,2,3]
indexSimulationRun = 1

for cancelProbsSetting in cancelProbsSetting_array:
    for meanNarrivalsSetting in meanNarrivalsSetting_array:

        print("cancelProbsSetting:",cancelProbsSetting)
        print("meanNarrivalsSetting:",meanNarrivalsSetting)
        
        # location of training_performance_metrics and location of where the output plots and model weights will be saved # TODO: change last digit
        filepath = "numExp_outputs_" + "OB" + repr(bumpingCostFactor) + "cnc" + repr(cancelProbsSetting) + "_fcArrivals" + repr(meanNarrivalsSetting) + "_" + repr(indexSimulationRun)
        # filepath = 'numExp_outputs_OB2cnc1_fcArrivals1_1' 

        with open(filepath + '/trRP_trLF_modelsTestRP_modelsTestLF.pkl','rb') as filePointer:  # Python 3: open(..., 'rb')
            # print(filePointer)       
            tr_RP, tr_LF, trModels_testPerf_meanRP, trModels_testPerf_meanLF = pickle.load(filePointer)


        rm_tr_RP = running_mean(tr_RP, 11) # rmInterval = 11
        diff_rm_tr_RP = [abs(rm_tr_RP[-1]-rm_tr_RP[-ind]) for ind in range(2,12)] 
        ##if max(diff_rm_tr_RP)<=0.005:
        ##    convergence_eps_005.append(iterInd*checkPointInterval) #TODO: SAVE convergence_eps
        if max(diff_rm_tr_RP)<=0.01:
            print("diff_rm_tr_RP<=0.01")
            #convergence_eps_01.append(iterInd*checkPointInterval) #TODO: SAVE convergence_eps
        ##if max(diff_rm_tr_RP)<=0.02:
        ##    convergence_eps_02.append(iterInd*checkPointInterval) #TODO: SAVE convergence_eps
        if max(diff_rm_tr_RP)<=0.05:
            # convergence_eps_05.append(iterInd*checkPointInterval) #TODO: SAVE convergence_eps
            print("diff_rm_tr_RP<=0.05")
        else:
            print("convergence not achieved")
