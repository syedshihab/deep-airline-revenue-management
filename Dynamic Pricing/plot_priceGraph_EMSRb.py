# TODO: may need to change filepath

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
from emsrbAgent_loggerClass_DP import *

indexSimulationRun = 1
outputDir = "EMSRbAgent_output_OB2cnc1fd1_2fp22pp"
if not os.path.isdir(outputDir):
    os.mkdir(outputDir)

##DQLagent_logger = infoLogger()
EMSRbagent_logger = infoLogger_emsrb()

# loading DQLagent_logger.pkl and EMSRbagent_logger.pkl

##with open(filepath + '/DQLagent_logger.pkl','rb') as f1:  # Python 3: open(..., 'rb')
##    DQLagent_logger.nBookings_fp, DQLagent_logger.nBookings_price, DQLagent_logger.preBumping_nBookings_fp, DQLagent_logger.preBumping_nBookings_price, DQLagent_logger.bumpingCost, \
##    DQLagent_logger.nCancellations_price, DQLagent_logger.episode_reward, DQLagent_logger.max_reward, DQLagent_logger.rewardPercentage, DQLagent_logger.overbooking, \
##    DQLagent_logger.loadFactor, DQLagent_logger.observations, DQLagent_logger.rewards, DQLagent_logger.actions = pickle.load(f1)


with open('emsrbAgent_testLogger_cnc1_fcArrivals1_2fp22pp_testedWith_true_mktSiml.pkl','rb') as f2:  # Python 3: open(..., 'rb')
    EMSRbagent_logger.nBookings_fp, EMSRbagent_logger.nBookings_price, EMSRbagent_logger.preBumping_nBookings_fp, EMSRbagent_logger.preBumping_nBookings_price, EMSRbagent_logger.bumpingCost, \
    EMSRbagent_logger.nCancellations_price, EMSRbagent_logger.episode_reward, EMSRbagent_logger.max_reward, EMSRbagent_logger.rewardPercentage, EMSRbagent_logger.overbooking, \
    EMSRbagent_logger.loadFactor, EMSRbagent_logger.observations, EMSRbagent_logger.rewards, EMSRbagent_logger.prices_fp, EMSRbagent_logger.openClose_WTPgrpFC, EMSRbagent_logger.BLemsrb = pickle.load(f2)

nTrEpisodes = 2000
rmInterval = 150

# Generating the revenue plot

nTimeSteps = 182
timeSteps = np.arange(nTimeSteps) # timeSteps = np.array([0,1,...,181])



testData = "testData_cnc1_fcArrivals1_2fp22pp_true_mktSiml" + ".mat"
##diff_eps_reward = np.array(DQLagent_logger.episode_reward) - np.array(EMSRbagent_logger.episode_reward)
##epsInd_maxDiff = np.where(diff_eps_reward==max(diff_eps_reward))[0][0]
##diff_epsReward_sorted = np.sort(diff_eps_reward)
##diff_epsReward_sorted_indices = np.argsort(diff_eps_reward)
epsIndex = 562 
nb_eps_steps = 182

fares_fp1 = []
fares_fp2 = []
# fares_fc3 = []

mat = scipy.io.loadmat(testData, squeeze_me=True)
priceMatrix = mat['priceMatrix']

##for action in DQLagent_logger.actions[epsIndex]:
##    fares_fp1.append(priceMatrix[action][0])
##    fares_fp2.append(priceMatrix[action][1])
##    # fares_fc3.append(priceMatrix[action][2])

emsrbAgent_fares_fp1 = []
emsrbAgent_fares_fp2 = []

for prices in EMSRbagent_logger.prices_fp[epsIndex]:
    emsrbAgent_fares_fp1.append(prices[0])
    emsrbAgent_fares_fp2.append(prices[1])

# fares_fp1 = priceMatrix[test_logger.actions[epsIndex]][0]

# Generating the pricing plot
nTimeSteps = nb_eps_steps
timeSteps = np.arange(nTimeSteps) # timeSteps = np.array([0,1,...,181])
##plt.plot(timeSteps, fares_fp1, label='DQL_fares_highFP')
##plt.plot(timeSteps, fares_fp2, label='DQL_fares_lowFP')
plt.plot(timeSteps, emsrbAgent_fares_fp1, '--', label='emsrb_fares_highFP')
plt.plot(timeSteps, emsrbAgent_fares_fp2, '--', label='emsrb_fares_lowFP')
plt.legend(loc='best')
# plt.title('dynamicPricingGraph_bestPerformance')
plt.ylabel('Fares ($)')
plt.xlabel('Time (day)')
plt.title('Fare product price evolution')
plt.savefig(outputDir + '/dynamicPricingGraph_epsInd_' + repr(epsIndex) + '.png')
plt.savefig(outputDir + '/dynamicPricingGraph_epsInd_' + repr(epsIndex) + '.pdf')
plt.close()
# plt.show()
