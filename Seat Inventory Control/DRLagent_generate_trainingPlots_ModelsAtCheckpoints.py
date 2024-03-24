# This script is used to generate test plots of optimalRewardPercentage and loadFactor of the best model (best among all the models saved at the checkpoints)

# TODO: specify bumpingCostFactor, cancelProbsSetting, meanNarrivalsSetting and indexSimulationRun; rmInterval
# inputs: bumpingCostFactor, cancelProbsSetting, meanNarrivalsSetting and indexSimulationRun; filepath, rmInterval
# outputs:

import pickle
import numpy as np
import matplotlib.pyplot as plt
from  DRLagent_armClasses_3fClasses_Tb1b2b3_SIC import running_mean

# Changing font size of all items (labels, ticks, etc.) in the plots
##font = {'size'   : 12} # 'serif':'Computer Modern Roman',  # Times New Roman
##plt.rc('font', **font)

bumpingCostFactor = 2
cancelProbsSetting = 1
meanNarrivalsSetting = 1
indexSimulationRun = 1

# location of training_performance_metrics and location of where the output plots and model weights will be saved # TODO: change last digit
filepath = "numExp_outputs_" + "OB" + repr(bumpingCostFactor) + "cnc" + repr(cancelProbsSetting) + "_fcArrivals" + repr(meanNarrivalsSetting) + "_" + repr(indexSimulationRun)


with open(filepath + '/trRP_trLF_modelsTestRP_modelsTestLF.pkl','rb') as f:  # Python 3: open(..., 'rb')
    # print(f)       
    tr_RP, tr_LF, trModels_testPerf_meanRP, trModels_testPerf_meanLF = pickle.load(f)

##with open(filepath + '/emsrbAgent_startTr6_testLogger.pkl','rb') as f2:  # Python 3: open(..., 'rb')
##    # print(f)
##    emsrbAgent_test_logger.seats_atEndEachEps, emsrbAgent_test_logger.nSeatsSoldByPrices, emsrbAgent_test_logger.nCancellationsByPrices, emsrbAgent_test_logger.episode_reward, \
##    emsrbAgent_test_logger.max_reward, emsrbAgent_test_logger.rewardPercentage, emsrbAgent_test_logger.overbooking, emsrbAgent_test_logger.loadFactor, emsrbAgent_test_logger.observations, \
##    emsrbAgent_test_logger.rewards, emsrbAgent_test_logger.prices, emsrbAgent_test_logger.BLemsrb = pickle.load(f2)

rewardPercentage = np.array(trModels_testPerf_meanRP)
loadFactor = np.array(trModels_testPerf_meanLF)


# Making plots of rP of trModels at the checkpoints 
nModels = len(trModels_testPerf_meanRP) # 200 # 300
rmInterval = 11 # 33 # 11

t1 = np.arange(nModels)+1
t2 = np.arange(1+(rmInterval-1)/2, nModels+1-(rmInterval-1)/2)

##testData_filename = 'training_c1_fd1_startTr6_test'
outputDir = filepath # "output_OB2cnc1fd1_21" # location where output plots and model weights are saved # TODO: change last digit
##mat = scipy.io.loadmat(testData_filename, squeeze_me=True)
##priceMatrix = mat['priceMatrix']


tnrfont = {'fontname':'Times New Roman'}

# plotting test_logger.rewardPercentage running mean
plt.plot(t2, running_mean(rewardPercentage, rmInterval), 'r')
plt.ylabel('Optimal Revenue (%)')#,**tnrfont)
plt.xlabel('Model')#,**tnrfont)
axes = plt.gca()
axes.set_xlim([0,nModels])
axes.set_ylim([70,100])
##plt.savefig(outputDir + '/trPlot_modelsAtCheckpoints_optimalRP_runningMean.eps')
plt.savefig(outputDir + '/trPlot_modelsAtCheckpoints_optimalRP_runningMean.pdf')
plt.close()
# plt.show()

### plotting test_logger.rewardPercentage + running mean
##plt.plot(t1, rewardPercentage, t2, running_mean(rewardPercentage, rmInterval), 'r')
##plt.ylabel('Optimal Revenue (%)')#,**tnrfont)
##plt.xlabel('Model')#,**tnrfont)
### axes = plt.gca()
###axes.set_xlim([1,nModels])
### axes.set_ylim([0,100])
##plt.savefig(outputDir + '/trPlot_modelsAtCheckpoints_optimalRP.eps')
##plt.savefig(outputDir + '/trPlot_modelsAtCheckpoints_optimalRP.pdf')
##plt.close()
### plt.show()

# plotting test_logger.loadFactor running mean
plt.plot(t2, running_mean(loadFactor, rmInterval), 'r')
plt.ylabel('Load Factor (%)') #,**tnrfont)
plt.xlabel('Model') #,**tnrfont)
axes = plt.gca()
axes.set_xlim([0,nModels])
axes.set_ylim([50,160])
##plt.savefig(outputDir + '/trPlot_modelsAtCheckpoints_LF_runningMean.eps')
plt.savefig(outputDir + '/trPlot_modelsAtCheckpoints_LF_runningMean.pdf')
plt.close()
# plt.show()

### plotting test_logger.loadFactor + running mean
##plt.plot(t1, loadFactor, t2, running_mean(loadFactor, rmInterval), 'r')
##plt.ylabel('Load Factor (%)') #,**tnrfont)
##plt.xlabel('Model') #,**tnrfont)
###axes = plt.gca()
###axes.set_xlim([1,nModels])
###axes.set_ylim([0,100])
##plt.savefig(outputDir + '/trPlot_modelsAtCheckpoints_LF.eps')
##plt.savefig(outputDir + '/trPlot_modelsAtCheckpoints_LF.pdf')
##plt.close()
### plt.show()
##
