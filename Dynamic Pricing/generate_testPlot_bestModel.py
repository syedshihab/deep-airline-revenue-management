# This script is used to generate test plots of optimalRewardPercentage and loadFactor of the best model (best among all the models saved at the checkpoints)

# TODO: may need to change filepath, testData_filename, epsMaxRPind, MATLABepisodeIndex
# TODO: For the case of 3 fare classes, uncomment lines with fares_fc3 

import pickle
import numpy as np

##import gym
##from gym import spaces
##import scipy.io
##import pdb
##import time 
##
##from keras.models import Sequential
##from keras.layers import Dense, Activation, Flatten
##from keras.optimizers import Adam
##from keras.callbacks import Callback
##from keras.utils import plot_model
##
##from rl.agents.dqn import DQNAgent
##from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy, GreedyQPolicy
##from rl.memory import SequentialMemory
##from rl.core import Processor, Env

import matplotlib.pyplot as plt
import os

# from armClasses_dynamicPricing import *
from armClasses_observationSpaceModified1_stateRewardScaled_2 import *
from emsrbAgent_loggerClass_DP import *



# Changing font size of all items (labels, ticks, etc.) in the plots
##font = {'size'   : 12} # 'serif':'Computer Modern Roman',  # Times New Roman
##plt.rc('font', **font)

filepath = 'output_OB2cnc1fd1_2fp22pp_twg1wg4c1c2_nHL3nHN256tmu1lr1pol123_bestAgent' # 'output_OB2cnc1fd1_55' # 'output_OB2cnc1fd1_53' # 'output_OB2cnc1fd1_11'

test_logger = infoLogger()

# loading DRLagent test logger
with open(filepath + '/testLogger_bestModel_trueMktSiml_lrNeg4_tmu10k.pkl','rb') as fp1:  # Python 3: open(..., 'rb')
    test_logger.nBookings_fp, test_logger.nBookings_price, test_logger.preBumping_nBookings_fp, test_logger.preBumping_nBookings_price, test_logger.bumpingCost, \
    test_logger.nCancellations_price, test_logger.episode_reward, test_logger.max_reward, test_logger.rewardPercentage, test_logger.overbooking, \
    test_logger.loadFactor, test_logger.observations, test_logger.rewards, test_logger.actions = pickle.load(fp1)


testLogger_EMSRb = infoLogger_emsrb()
# loading EMSRbagent test logger
with open('emsrbAgent_testLogger_cnc1_fcArrivals1_2fp22pp_testedWith_true_mktSiml2.pkl','rb') as fp2:  # Python 3: open(..., 'rb')
    testLogger_EMSRb.nBookings_fp, testLogger_EMSRb.nBookings_price, testLogger_EMSRb.preBumping_nBookings_fp, testLogger_EMSRb.preBumping_nBookings_price, testLogger_EMSRb.bumpingCost, testLogger_EMSRb.nCancellations_price, \
    testLogger_EMSRb.episode_reward, testLogger_EMSRb.max_reward, testLogger_EMSRb.rewardPercentage, testLogger_EMSRb.overbooking, testLogger_EMSRb.loadFactor, testLogger_EMSRb.observations, testLogger_EMSRb.rewards, testLogger_EMSRb.prices_fp, \
    testLogger_EMSRb.openClose_WTPgrpFC, testLogger_EMSRb.BLemsrb = pickle.load(fp2)

##with open(filepath + '/emsrbAgent_startTr6_testLogger.pkl','rb') as f2:  # Python 3: open(..., 'rb')
##    # print(f)
##    emsrbAgent_test_logger.seats_atEndEachEps, emsrbAgent_test_logger.nSeatsSoldByPrices, emsrbAgent_test_logger.nCancellationsByPrices, emsrbAgent_test_logger.episode_reward, \
##    emsrbAgent_test_logger.max_reward, emsrbAgent_test_logger.rewardPercentage, emsrbAgent_test_logger.overbooking, emsrbAgent_test_logger.loadFactor, emsrbAgent_test_logger.observations, \
##    emsrbAgent_test_logger.rewards, emsrbAgent_test_logger.prices, emsrbAgent_test_logger.BLemsrb = pickle.load(f2)


RP = np.array(test_logger.rewardPercentage)*100
LF = np.array(test_logger.loadFactor)*100

RP_EMSRb = np.array(testLogger_EMSRb.rewardPercentage)*100
LF_EMSRb = np.array(testLogger_EMSRb.loadFactor)*100


# Making plots of rP of trModels at the checkpoints 
nTestEpisodes = 2000 # 300
rmInterval = 33 # 11

t1 = np.arange(nTestEpisodes)+1
t2 = np.arange(1+(rmInterval-1)/2, nTestEpisodes+1-(rmInterval-1)/2)

##testData_filename = 'training_c1_fd1_startTr6_test'
outputDir = filepath # "output_OB2cnc1fd1_21" # location where output plots and model weights are saved # TODO: change last digit
##mat = scipy.io.loadmat(testData_filename, squeeze_me=True)
##priceMatrix = mat['priceMatrix']


tnrfont = {'fontname':'Times New Roman'}

### Plotting RP + running mean
##plt.plot(t1, RP, t2, running_mean(RP, rmInterval), 'r')
##plt.ylabel('Optimal revenue (%)')#,**tnrfont)
##plt.xlabel('Episode')#,**tnrfont)
##axes = plt.gca()
###axes.set_xlim([1,nTestEpisodes])
##axes.set_ylim([0,100])
##plt.savefig(outputDir + '/testPlot_RP_trueMktSiml.pdf')
##plt.savefig(outputDir + '/testPlot_RP_trueMktSiml.png')
##plt.close()
### plt.show()

# Plotting RP running mean
plt.plot(t2, running_mean(RP, rmInterval), 'r', label='Moving average')
plt.axhline(y=np.mean(RP), color='b', linestyle='--', lw=1, label='Overall average')
plt.ylabel('Optimal revenue (%)')#,**tnrfont)
plt.xlabel('Episode')#,**tnrfont)
axes = plt.gca()
#axes.set_xlim([1,nTestEpisodes])
axes.set_ylim([50,100])
plt.legend()
# plt.show()
plt.savefig(outputDir + '/testPlot_RP_runningMean_trueMktSiml_2.pdf')
plt.savefig(outputDir + '/testPlot_RP_runningMean_trueMktSiml_2.png')
plt.close()
# plt.show()


### Plotting LF + running mean
##plt.plot(t1, LF, t2, running_mean(LF, rmInterval), 'r')
##plt.ylabel('Load factor (%)') #,**tnrfont)
##plt.xlabel('Episode') #,**tnrfont)
##axes = plt.gca()
###axes.set_xlim([1,nTestEpisodes])
##axes.set_ylim([0,100])
##plt.savefig(outputDir + '/testPlot_LF_trueMktSiml.pdf')
##plt.savefig(outputDir + '/testPlot_LF_trueMktSiml.png')
##plt.close()
### plt.show()

# Plotting LF running mean
plt.plot(t2, running_mean(LF, rmInterval), 'r', label='Moving average')
plt.axhline(y=np.mean(LF), color='b', linestyle='--', lw=1, label='Overall average')
plt.ylabel('Load factor (%)') #,**tnrfont)
plt.xlabel('Episode') #,**tnrfont)
axes = plt.gca()
#axes.set_xlim([1,nTestEpisodes])
axes.set_ylim([20,120])
plt.legend()
plt.savefig(outputDir + '/testPlot_LF_runningMean_trueMktSiml_2.pdf')
plt.savefig(outputDir + '/testPlot_LF_runningMean_trueMktSiml_2.png')
plt.close()
# plt.show()

