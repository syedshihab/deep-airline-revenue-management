# This script is used to generate test plots of optimalRewardPercentage and loadFactor of the best model (best among all the models saved at the checkpoints)

# Input parameters: bumpingCostFactor, cancelProbsSetting, meanNarrivalsSetting, indexSimulationRun
# specify filepath and filename of the pickle object of the best model's testLogger


import pickle

import numpy as np
import matplotlib.pyplot as plt
# from statistics import mean # mean function is not giving an accurate result for reason; it is giving a rounded result;
# so using np.mean instead to calculate the averages; sum(list)/len(list) also gives the accurate result;


# from armClasses_dynamicPricing import *
# from  DRLagent_armClasses_3fClasses_Tb1b2b3_SIC import infoLogger,running_mean
from DRLagent_essentialClassesFunctions import * 


# Changing font size of all items (labels, ticks, etc.) in the plots
##font = {'size'   : 12} # 'serif':'Computer Modern Roman',  # Times New Roman
##plt.rc('font', **font)

bumpingCostFactor = 2
cancelProbsSetting = 1
meanNarrivalsSetting = 1
indexSimulationRun = 1

# location of testLogger_bestModel_500
filepath = "numExp_outputs_" + "OB" + repr(bumpingCostFactor) + "cnc" + repr(cancelProbsSetting) + "_fcArrivals" + repr(meanNarrivalsSetting) + "_" + repr(indexSimulationRun)
# filepath = 'numExp_outputs_OB2cnc1_fcArrivals1_1' 

test_logger = infoLogger()



with open(filepath + '/testLogger_bestModel_500.pkl','rb') as filePointer:  # Python 3: open(..., 'rb')
    # print(filePointer)       
    test_logger.nBookings, test_logger.nCancellations, test_logger.episode_reward, test_logger.max_reward, test_logger.rewardPercentage, \
    test_logger.overbooking, test_logger.loadFactor, test_logger.observations, test_logger.rewards, test_logger.actions = pickle.load(filePointer)


nBookings_eps_fClass = test_logger.nBookings # list of arrays; element accessing syntax: nBookings_eps_fClass[eps][fClass]
##test_logger.loadFactor = np.array(test_logger.loadFactor)*100

nBookingsFclass0_eps = [item[0] for item in nBookings_eps_fClass]
nBookingsFclass1_eps = [item[1] for item in nBookings_eps_fClass]
nBookingsFclass2_eps = [item[2] for item in nBookings_eps_fClass] # after bumping
beforeBumping_nBookingsFclass2_eps = [nBookingsFclass2_eps[i] + test_logger.overbooking[i] for i in range(len(nBookingsFclass2_eps))] # before bumping

totalNbookings_eps = [nBookingsFclass0_eps[i] + nBookingsFclass1_eps[i] + beforeBumping_nBookingsFclass2_eps[i] for i in range(len(nBookingsFclass0_eps))]
avg_totalNbookings = np.mean(np.array(totalNbookings_eps))

# Making plots of rP of trModels at the checkpoints 

nTestEpisodes = 300
# rmInterval = 33 # 11

t1 = np.arange(nTestEpisodes)+1
##t2 = np.arange(1+(rmInterval-1)/2, nTestEpisodes+1-(rmInterval-1)/2)

##testData_filename = 'training_c1_fd1_startTr6_test'
outputDir = filepath # "output_OB2cnc1fd1_21" # location where output plots and model weights are saved # TODO: change last digit
##mat = scipy.io.loadmat(testData_filename, squeeze_me=True)
##priceMatrix = mat['priceMatrix']


tnrfont = {'fontname':'Times New Roman'}

# plotting test_logger.nBookings + meanNbookings lines for each fare class
plt.plot(t1, nBookingsFclass0_eps, label='Fare class H', color='black', linestyle=':', lw=0.5)
plt.axhline(y=np.mean(np.array(nBookingsFclass0_eps)), color='b', linestyle='--', lw=1)
plt.plot(t1, nBookingsFclass1_eps, label='Fare class M', color='c', linestyle='--', lw=1) # forestgreen; limegreen; green; lime
plt.axhline(y=np.mean(np.array(nBookingsFclass1_eps)), color='b', linestyle='--', lw=1)
plt.plot(t1, beforeBumping_nBookingsFclass2_eps, label='Fare class L', color='lightcoral', linestyle='-')
plt.axhline(y=np.mean(np.array(beforeBumping_nBookingsFclass2_eps)), color='b', linestyle='--', lw=1)
##plt.legend(loc=(0.02,0.2))
plt.legend()
plt.ylabel('Number of bookings')#,**tnrfont)
plt.xlabel('Episode')#,**tnrfont)
axes = plt.gca()
axes.set_xlim([0,nTestEpisodes])
axes.set_ylim([0,100])
# plt.yticks(list(plt.yticks()[0]) + [np.mean(test_logger.rewardPercentage)]) # adding an extra y-tick at the average value
# plt.show()
# plt.savefig(outputDir + '/testPlot_bestModel_optimalRP_runningMean.eps')

##plt.show()
plt.savefig(outputDir + '/testPlot_bestModel_bookingPlot2.pdf')
plt.close()


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
##plt.axhline(y=np.mean(test_logger.loadFactor), color='b', linestyle='--', lw=1)
##plt.ylabel('Load Factor (%)') #,**tnrfont)
##plt.xlabel('Episode') #,**tnrfont)
##axes = plt.gca()
##axes.set_xlim([0,nTestEpisodes])
##axes.set_ylim([80,130])
### plt.yticks(list(plt.yticks()[0]) + [np.mean(test_logger.loadFactor)]) # adding an extra y-tick at the average value
####plt.savefig(outputDir + '/testPlot_bestModel_LF_runningMean.eps')
##plt.savefig(outputDir + '/testPlot_bestModel_LF_runningMean.pdf')
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
