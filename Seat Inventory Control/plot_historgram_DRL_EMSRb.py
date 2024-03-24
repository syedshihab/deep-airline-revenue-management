# This script is used to generate test plots of optimalRewardPercentage and loadFactor of the best model (best among all the models saved at the checkpoints)

# TODO: may need to change filepath, testData_filename, epsMaxRPind, MATLABepisodeIndex
# TODO: For the case of 3 fare classes, uncomment lines with fares_fc3 

import pickle

import numpy as np
import matplotlib.pyplot as plt


# from armClasses_dynamicPricing import *
from  DRLagent_armClasses_3fClasses_Tb1b2b3_SIC import infoLogger,running_mean


# Changing font size of all items (labels, ticks, etc.) in the plots
##font = {'size'   : 12} # 'serif':'Computer Modern Roman',  # Times New Roman
##plt.rc('font', **font)

bumpingCostFactor = 2
cancelProbsSetting = 1
meanNarrivalsSetting = 1
indexSimulationRun = 1

# location of training_performance_metrics and location of where the output plots and model weights will be saved # TODO: change last digit
filepath = "numExp_outputs_" + "OB" + repr(bumpingCostFactor) + "cnc" + repr(cancelProbsSetting) + "_fcArrivals" + repr(meanNarrivalsSetting) + "_" + repr(indexSimulationRun)
# filepath = 'numExp_outputs_OB2cnc1_fcArrivals1_1' 

test_logger = infoLogger()

# Getting back the objects:
##with open(filepath + '/trainLogger.pkl','rb') as f:  # Python 3: open(..., 'rb')
##    # print(f)
##    info_logger.seats, info_logger.nSeatsSoldByPrices, info_logger.nCancellationsByPrices, info_logger.episode_reward, \
##    info_logger.max_reward, info_logger.rewardPercentage, info_logger.overbooking, info_logger.loadFactor, info_logger.observations, \
##    info_logger.rewards, info_logger.actions = pickle.load(f)

with open(filepath + '/testLogger_bestModel_500.pkl','rb') as filePointer:  # Python 3: open(..., 'rb')
    # print(filePointer)       
    test_logger.nBookings, test_logger.nCancellations, test_logger.episode_reward, test_logger.max_reward, test_logger.rewardPercentage, \
    test_logger.overbooking, test_logger.loadFactor, test_logger.observations, test_logger.rewards, test_logger.actions = pickle.load(filePointer)


##with open(filepath + '/emsrbAgent_startTr6_testLogger.pkl','rb') as f2:  # Python 3: open(..., 'rb')
##    # print(f)
##    emsrbAgent_test_logger.seats_atEndEachEps, emsrbAgent_test_logger.nSeatsSoldByPrices, emsrbAgent_test_logger.nCancellationsByPrices, emsrbAgent_test_logger.episode_reward, \
##    emsrbAgent_test_logger.max_reward, emsrbAgent_test_logger.rewardPercentage, emsrbAgent_test_logger.overbooking, emsrbAgent_test_logger.loadFactor, emsrbAgent_test_logger.observations, \
##    emsrbAgent_test_logger.rewards, emsrbAgent_test_logger.prices, emsrbAgent_test_logger.BLemsrb = pickle.load(f2)

RP = np.array(test_logger.rewardPercentage)*100
LF = np.array(test_logger.loadFactor)*100

plt.hist(RP)
plt.show()

with open(filepath + '/testLogger_bestModel_500.pkl','rb') as filePointer:  # Python 3: open(..., 'rb')
    # print(filePointer)       
    test_logger.nBookings, test_logger.nCancellations, test_logger.episode_reward, test_logger.max_reward, test_logger.rewardPercentage, \
    test_logger.overbooking, test_logger.loadFactor, test_logger.observations, test_logger.rewards, test_logger.actions = pickle.load(filePointer)



##
### Making plots of rP of trModels at the checkpoints 
##nTestEpisodes = 300
##rmInterval = 33 # 11
##
##t1 = np.arange(nTestEpisodes)+1
##t2 = np.arange(1+(rmInterval-1)/2, nTestEpisodes+1-(rmInterval-1)/2)
##
####testData_filename = 'training_c1_fd1_startTr6_test'
##outputDir = filepath # "output_OB2cnc1fd1_21" # location where output plots and model weights are saved # TODO: change last digit
####mat = scipy.io.loadmat(testData_filename, squeeze_me=True)
####priceMatrix = mat['priceMatrix']
##
##
##tnrfont = {'fontname':'Times New Roman'}
##
### plotting test_logger.rewardPercentage running mean only
##plt.plot(t2, running_mean(test_logger.rewardPercentage, rmInterval), 'r')
##plt.axhline(y=np.mean(test_logger.rewardPercentage), color='b', linestyle='--', lw=1, label="Overall average")
##plt.ylabel('Optimal Revenue (%)')#,**tnrfont)
##plt.xlabel('Episode')#,**tnrfont)
##axes = plt.gca()
##axes.set_xlim([0,nTestEpisodes])
##axes.set_ylim([85,105])
##plt.legend()
### plt.yticks(list(plt.yticks()[0]) + [np.mean(test_logger.rewardPercentage)]) # adding an extra y-tick at the average value
### plt.show()
### plt.savefig(outputDir + '/testPlot_bestModel_optimalRP_runningMean.eps')
##plt.savefig(outputDir + '/testPlot_bestModel_optimalRP_runningMean_3.pdf')
##plt.close()
##
##
##### olotting test_logger.rewardPercentage + running mean
####plt.plot(t1, test_logger.rewardPercentage, t2, running_mean(test_logger.rewardPercentage, rmInterval), 'r')
####plt.ylabel('Optimal Revenue (%)')#,**tnrfont)
####plt.xlabel('Episode')#,**tnrfont)
##### axes = plt.gca()
#####axes.set_xlim([1,nTestEpisodes])
##### axes.set_ylim([0,100])
####plt.savefig(outputDir + '/testPlot_bestModel_optimalRP.eps')
####plt.savefig(outputDir + '/testPlot_bestModel_optimalRP.pdf')
####plt.close()
##### plt.show()
##
### olotting test_logger.loadFactor running mean
##plt.plot(t2, running_mean(test_logger.loadFactor, rmInterval), 'r')
##plt.axhline(y=np.mean(test_logger.loadFactor), color='b', linestyle='--', lw=1, label="Overall average")
##plt.ylabel('Load Factor (%)') #,**tnrfont)
##plt.xlabel('Episode') #,**tnrfont)
##axes = plt.gca()
##axes.set_xlim([0,nTestEpisodes])
##axes.set_ylim([80,130])
##plt.legend()
### plt.yticks(list(plt.yticks()[0]) + [np.mean(test_logger.loadFactor)]) # adding an extra y-tick at the average value
####plt.savefig(outputDir + '/testPlot_bestModel_LF_runningMean.eps')
##plt.savefig(outputDir + '/testPlot_bestModel_LF_runningMean_3.pdf')
##plt.close()
### plt.show()
##
##### olotting test_logger.loadFactor + running mean
####plt.plot(t1, test_logger.loadFactor, t2, running_mean(test_logger.loadFactor, rmInterval), 'r')
####plt.ylabel('Load Factor (%)') #,**tnrfont)
####plt.xlabel('Episode') #,**tnrfont)
#####axes = plt.gca()
#####axes.set_xlim([1,nTestEpisodes])
#####axes.set_ylim([0,100])
####plt.savefig(outputDir + '/testPlot_bestModel_LF.eps')
####plt.savefig(outputDir + '/testPlot_bestModel_LF.pdf')
####plt.close()
##### plt.show()
####
