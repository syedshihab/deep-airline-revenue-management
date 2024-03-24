# this script is used to plot price(t) set by the best model

# TODO: may need to change filepath, testData_filename, epsMaxRPind, MATLABepisodeIndex
# TODO: For the case of 3 fare classes, uncomment lines with fares_fc3 

import pickle
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os

from  armClasses_observationSpaceModified1_stateRewardScaled_2 import *

filepath = "output_" + "OB" + repr(2) + "cnc" + repr(1) + "fd" + repr(1) + "_" + "2fp22pp_twg1wg4c1c2" + "_" + repr(1000)

test_logger = infoLogger()

with open(filepath + '/testLogger_bestModel.pkl','rb') as filepointer1:  # Python 3: open(..., 'rb')
    # print(f)
    test_logger.nBookings_fp, test_logger.nBookings_price, test_logger.preBumping_nBookings_fp, test_logger.preBumping_nBookings_price, test_logger.bumpingCost, \
    test_logger.nCancellations_price, test_logger.episode_reward, test_logger.max_reward, test_logger.rewardPercentage, test_logger.overbooking, \
    test_logger.loadFactor, test_logger.observations, test_logger.rewards, test_logger.actions = pickle.load(filepointer1)
    

##with open(filepath + '/emsrbAgent_startTr6_testLogger.pkl','rb') as f2:  # Python 3: open(..., 'rb')
##    # print(f)
##    emsrbAgent_test_logger.seats_atEndEachEps, emsrbAgent_test_logger.nSeatsSoldByPrices, emsrbAgent_test_logger.nCancellationsByPrices, emsrbAgent_test_logger.episode_reward, \
##    emsrbAgent_test_logger.max_reward, emsrbAgent_test_logger.rewardPercentage, emsrbAgent_test_logger.overbooking, emsrbAgent_test_logger.loadFactor, emsrbAgent_test_logger.observations, \
##    emsrbAgent_test_logger.rewards, emsrbAgent_test_logger.prices, emsrbAgent_test_logger.BLemsrb = pickle.load(f2)


testData_filename = "trData_cnc" + repr(1) + "_fcArrivals" + repr(1) + "_2fp22pp_mktSiml_mktEst" + ".mat"
outputDir = filepath # "output_OB2cnc1fd1_21" # location where output plots and model weights are saved # TODO: change last digit
mat = scipy.io.loadmat(testData_filename, squeeze_me=True)
priceMatrix = mat['priceMatrix']


# identifying the episode in which the agent achieved the highest rewardPercentage

epsMaxRPind = test_logger.rewardPercentage.index(max(test_logger.rewardPercentage))
# epsMaxRPind = 2 # {0,1,2,...nTestEpisodes}

print('episode index (counting from 0) =',epsMaxRPind)
print('episode rewardPercentage =',test_logger.rewardPercentage[epsMaxRPind])
print('episode loadFactor =',test_logger.loadFactor[epsMaxRPind])

fares_fc1 = []
fares_fc2 = []
# fares_fc3 = []

for action in test_logger.actions[epsMaxRPind]:
    fares_fc1.append(priceMatrix[action][0])
    fares_fc2.append(priceMatrix[action][1])
    # fares_fc3.append(priceMatrix[action][2])

##emsrbAgent_fares_fc1 = []
##emsrbAgent_fares_fc2 = []   

# fares_fc1 = priceMatrix[test_logger.actions[epsMaxRPind]][0]

# Generating the pricing plot
nTimeSteps = 182
t1 = np.arange(nTimeSteps) # t1 = np.array([0,1,...,181])
plt.plot(t1, fares_fc1, t1, fares_fc2) #, t1, fares_fc3) 
# plt.title('dynamicPricingGraph_bestPerformance')
plt.ylabel('Fares ($)')
plt.xlabel('Time (day)')
plt.title('Fare class price evolution')
plt.savefig(outputDir + '/dynamicPricingGraph_bestPerformanceDQL.pdf')
plt.close()
# plt.show()


# Generating the bar plot for nArrivals of different wtpGroups

# determining the number of arrivals of different wtpGroups at all time steps in the best episode of the agent
trDataPy = mat['trDataPy']
nArrivals = mat['nArrivals']
nTrEpisodes = 100000 # TODO: adjust if necessary
MATLABepisodeIndex = epsMaxRPind + 22000 + 1 # TODO: adjust if necessary; add +1???
nTotalPax = trDataPy[MATLABepisodeIndex].shape[0] # trDataPy[nTrEpisodes + epsMaxRPind].shape[0]
currentEpisode = trDataPy[MATLABepisodeIndex-1] # trDataPy[nTrEpisodes + epsMaxRPind] # 10000 + 77
nArrivals_epsMaxRPind_wtpGrp_t = nArrivals[:,:,MATLABepisodeIndex]


##nArrivals_wtpGroups = {}
##
##for ind in range(4):    
##    nArrivals_wtpGroups[ind]=[0] * 182 # nArrivals_wtpGroups[ind] = [0, 0, ... , 0] = zeros(1,182)
##
##paxIndex = 0
##
##for time in range(182):
##    arvTime = currentEpisode[paxIndex,0]
##    while (arvTime==time): # pax arrives within this time step    
##        wtpGroup = int(currentEpisode[paxIndex,1]) # wtpGroup = {0,1,2,3,4,5}
##        # fclass = fclassIdentifierPy[wtpGroup] # fclassIdentifierPy = {0,1,2}
##        # cncTime = currentEpisode[paxIndex,2]
##        # TODO: cancellations not considered
##
##        nArrivals_wtpGroups[wtpGroup][time]+=1
##        
##        if(paxIndex < nTotalPax - 1):
##            paxIndex += 1                
##            arvTime = currentEpisode[paxIndex,0]
##        else:                
##            arvTime = -2 # break while loop # break
####            time = 182 # break for loop
##
##    if (arvTime == -2):
##        break


##plt.plot(t1, nArrivals_wtpGroups[0], t1, nArrivals_wtpGroups[1], t1, nArrivals_wtpGroups[2],
##         t1, nArrivals_wtpGroups[3], t1, nArrivals_wtpGroups[4], t1, nArrivals_wtpGroups[5])
### plt.title('dynamicPricingGraph_bestPerformance')
##plt.ylabel('Number of arrivals')
##plt.xlabel('Time (day)')
##plt.savefig(outputDir + '/nArrivals_wtpGroups.png')
##plt.close()
##
####nArrivals_wtpGroups[4][timeStep]
####nArrivals_wtpGroups[5][timeStep]
##
####for x in range(6):
####    print("nArrivals_wtpGroups_{}: {}".format(x,nArrivals_wtpGroups[x][179]))
##
### TODO: analyze how the agent is varying the fares of the fare classes? Is it taking into consideration time remaining, seats available (adjustment to stochastic demand) and late arrival
### of price-insensitive business passengers with higher wtp?
##
##plt.bar(t1, nArrivals_wtpGroups[0], width=1, align='edge')
### plt.xticks(y_pos, objects)
##plt.ylabel('Number of arrivals')
##plt.xlabel('Time (day)')
##plt.title('Number of arrivals of passengers of different wtpGroups across time in the best episode of the agent')
##plt.show()
##plt.savefig(outputDir + '/nArrivals_wtpGroups.png')
##plt.close()

#
# data to plot

plt.subplot(211)
plt.plot(t1, fares_fc1, label='high fare class')
plt.plot(t1, fares_fc2, label='low fare class') # , t1, fares_fc3)
# plt.title('dynamicPricingGraph_bestPerformance')
##plt.legend(loc='best')
plt.legend(loc=(0.02,0.2))
plt.ylabel('Fares ($)')
plt.xlabel('Time (day)')
# plt.title('Fare class price evolution')

plt.subplot(212)
# create plot
# fig, ax = plt.subplots()

bar_width = 1/6
opacity = 0.8

rects0 = plt.bar(t1, nArrivals_epsMaxRPind_wtpGrp_t[0], bar_width, align='edge',
alpha=opacity, label='WTP = $600')

rects1 = plt.bar(t1 + bar_width, nArrivals_epsMaxRPind_wtpGrp_t[1], bar_width, align='edge',
alpha=opacity, label='WTP = $400')

rects2 = plt.bar(t1 + 2*bar_width, nArrivals_epsMaxRPind_wtpGrp_t[2], bar_width, align='edge',
alpha=opacity, label='WTP = $200')

rects3 = plt.bar(t1 + 3*bar_width, nArrivals_epsMaxRPind_wtpGrp_t[3], bar_width, align='edge',
alpha=opacity, label='WTP = $100')

##rects4 = plt.bar(t1 + 4*bar_width, nArrivals_wtpGroups[4], bar_width, align='edge',
##alpha=opacity, label='wtpGroup4')
##
##rects5 = plt.bar(t1 + 5*bar_width, nArrivals_wtpGroups[5], bar_width, align='edge',
##alpha=opacity, label='wtpGroup5')

plt.xlabel('Time (day)')
plt.ylabel('Number of arrivals')
# plt.title('Number of arrivals of passengers of different wtpGroups across time in the best episode of the agent')
# plt.xticks(index + bar_width, ('A', 'B', 'C', 'D'))
plt.legend()
plt.tight_layout()
plt.savefig(outputDir + '/fareEvolution_paxArrival_plots.pdf')
##plt.show()

