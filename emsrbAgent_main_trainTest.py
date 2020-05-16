# This script is used to test and record the performance of an EMSRb agent in the ARM problem
# Author: Syed A.M. Shihab

import pickle
import sys
import numpy as np
import scipy.io

from revpy.revpy import protection_levels
from revpy.helpers import cumulative_booking_limits
from revpy.fare_transformation import calc_fare_transformation

from emsrbAgent_loggerClass_SIC import *

maxBumpingCost = 6.75
bcfArray = [2] # [2.0, 2.5]
cncArray = [1,2,3] # [4] # [1] # [ 2 3]
fdArray = [1,2,3] # [4] # [3] # [1]
nTestEps = 500 # 150


f = open("EMSRbAgentResults.txt", "w")
f.write("bumpingCostFactor, cancelProbsSetting, meanNarrivalsSetting, mean test RP, mean test LF" + "\n")

for bumpingCostFactor in bcfArray:
    for cancelProbsSetting in cncArray:
        for meanNarrivalsSetting in fdArray:

            print('cancelProbsSetting =', cancelProbsSetting)
            print('meanNarrivalsSetting =', meanNarrivalsSetting)
                               
            testData = "testData_cnc" + repr(cancelProbsSetting) + "_fcArrivals" + repr(meanNarrivalsSetting) + ".mat"            

            mat = scipy.io.loadmat(testData, squeeze_me=True)
            
            nFareClasses = mat['nFareClasses']            
            capacity = mat['capacity']
            totalTime = mat['bookingHorizon']                         
            fclassIdentifierPy = mat['fclassIdentifierPy']            
            cncFee = mat['cncFee']
            cncProb_fClass_t = mat['cncProb_fClasses']
            cncProb_fClass_t = np.vstack ((cncProb_fClass_t, np.array([[0]*182]))) 
            fClassPrices = mat['fClassPrices']
            trDataPy = mat['trDataPy']
            nArrivals_by_eps_fClass = mat['nArrivals_by_eps_fClass']
            max_reward_list = mat['maxReward'] 
            
            nCancellations = np.zeros((nFareClasses-1), dtype=int) # nCancellations = ([0,0]);
            cancellations = []

            done = False

            fClassMeanArrivals = mat['fClassMeanArrivals']
            lambdaValues = mat['lambdaValues']           
             

            testLogger = infoLogger_emsrb()

            for currentEpisodeIndex in range(nTestEps):

                # new episode
                testLogger.on_episode_begin(currentEpisodeIndex)
                
                # reseting variables and loading episode info
                currentEpisode = trDataPy[currentEpisodeIndex]
                episode_reward = 0
                max_reward = max_reward_list[currentEpisodeIndex] # theoretical upper bound of revenue/theoretical optimal revenue                
                nSeatsAvailable = capacity # All seats are available for booking at time=0
                                
                paxIndex = 0
                nTotalPax = currentEpisode.shape[0] # = nArrivalsCurrentEps[0] + nArrivalsCurrentEps[1] + nArrivalsCurrentEps[2]
                
                nBookings_fClass = np.zeros(nFareClasses, dtype=int) # nBookings_fClass = [0 0 0]                
                nCancellations = np.zeros(nFareClasses-1, dtype=int) 

                cancellations = [] # empty list

                done = False 
                
                
                for time in range(totalTime): 
                    # new step
                    
                    timeRem = totalTime - time
                    
                    # determining booking limits using the EMSRb method
                    # deteriming demand and cancellation forecasts of each fare class for the remaining time of the booking period
                    demandForecast_fClass = np.zeros((nFareClasses))
                    cncForecast_fClass = np.zeros((nFareClasses))
                    # At any given time, demand forecast of each fare class for the remaining time period depends on the expected number of future arrivals and cancellations at each subsequent time step                    
                    for fClassInd in range(nFareClasses):
                        changeInB = 0 # this variable stores the expected changeInB at time = timeInd
                        b_fClass_t = nBookings_fClass[fClassInd] # Final value of this variable = expected b_fClass_182
                        nCnc = 0
                        nArv = 0
                        for timeInd in range(time,totalTime):
                            nCnc = nCnc + b_fClass_t*cncProb_fClass_t[fClassInd,timeInd]                            
                            changeInB = -b_fClass_t*cncProb_fClass_t[fClassInd,timeInd]+lambdaValues[fClassInd,timeInd] # alternative equation: changeInB = changeInB - b_fClass_t*cncProb_fClass_t[fClassInd,timeInd]+lambdaValues[fClassInd,timeInd]
                            b_fClass_t = b_fClass_t + changeInB # alternative equation: b_fClass_t = nBookings_fClass[fClassInd] + changeInB
                        # nArv = np.sum(lambdaValues[fClassInd,time:]) # expected number of future arrivals
                        demandForecast_fClass[fClassInd] = np.sum(lambdaValues[fClassInd,time:]) # alternative equation: demandForecast_fClass[fClassInd] = changeInB
                        cncForecast_fClass[fClassInd] = nCnc

                                        
                    paxCount = np.zeros(3, dtype=int)
                    
                    AUcapacity = int(nSeatsAvailable + np.sum(cncForecast_fClass))

                    p = protection_levels(fClassPrices, demandForecast_fClass, method='EMSRb')        
                
                    BLemsrb = cumulative_booking_limits(p, AUcapacity) # proxy for the action of closing fare classes; bookingLimits varies from flight episode to episode based on stochastic demand

                    fClassOpen = [0,0,0]
                    for fClass in range(nFareClasses):
                        if BLemsrb[fClass]>0:
                            fClassOpen[fClass]=1
                                        
                    

                    reward = 0 # step reward
                    # Checking if there are pax arriving at this time step                               
                    # action = {0,1,2}                                                        
                    arvTime = currentEpisode[paxIndex,0]
                    while (arvTime==time): # pax arrives within this time step

                        fclass = int(currentEpisode[paxIndex,1]) # fclass = {0,1,2}
                        cncTime = currentEpisode[paxIndex,2]                       
                        # pax can make a booking if the fClass they are seeking is open and has not exceeded its booking limit within the given time step
                        if (fClassOpen[fclass]==1):                                                         
                            # checking if bookingLimit > paxCount; bookingLimit = 0 means that the fareClass is closed
                            if BLemsrb[fclass]>paxCount[fclass]:
                                paxCount[fclass] += 1 # fclass = fare class of passenger who booked the ticket = {0,1,2}
                                nBookings_fClass[fclass] += 1 # fclass = fare class of passenger who booked the ticket = {0,1,2}
                                farePaid = fClassPrices[fclass]                                 
                                # Computing immediate reward
                                reward += farePaid
                                
                                # Checking if passenger will cancel reservation later on
                                if (cncTime > -1):                        
                                    cancellations.append((cncTime, fclass, farePaid))
                                    # sort on first index (cancellation time)
                                    cancellations.sort(key= lambda elem: elem[0])
                                    
                      
                        # Move on to next pax
                        if (paxIndex < nTotalPax - 1):
                            paxIndex += 1                
                            arvTime = currentEpisode[paxIndex,0]
                        else:                
                            arvTime = -1 # break while loop 

                    # removing any cancellations that occur during this time step
                    
                    while((len(cancellations) > 0) and (cancellations[0][0] == time)):
                        fclassCnc = cancellations[0][1]
                        farePaid = cancellations[0][2]            
                        nBookings_fClass[fclassCnc] -= 1                        
                        nCancellations[fclassCnc] += 1 # nCancellationsByPrices
                        # paxCount[fareInd] -= 1
                        
                        # Computing reward
                        reward -= (farePaid - cncFee[fclassCnc]) # fClassCancelRefund[fclassCnc]*fclassPrices[fclassCnc]            
                        # Removing first element of the stack
                        cancellations.pop(0)
                        

                            
                    if (time==totalTime-1): # if time == 181
                        done = True
                        
                ##        else:
                ##            time = time + 1 # time advances one step in the episode
                                        
                    if (done):            
                        overbooking = 0
                        if(sum(nBookings_fClass) > capacity): # Oversold/overbooked flight; need to bump some passengers
##                            print('Oversold/overbooked flight')
##                            print('currentEpisodeIndex (starting index is zero) =',currentEpisodeIndex)
                            nPaxToBump = sum(nBookings_fClass) - capacity # number of passengers that need to voluntarily/involuntarily denied boarding
                            overbooking = nPaxToBump
##                            print('nPaxToBumped =',nPaxToBump)
                            
                            if(nPaxToBump <= nBookings_fClass[nFareClasses-1]): # if(nPaxToBump <= nBookings_fClass[3]):
##                                print('bumping low fare class passengers')
                                nBookings_fClass[nFareClasses-1] -= nPaxToBump                    
                                # Assuming the scheduled flight is a domestic one and bumped pax will experience 1 to 2 hour arrival delay
                                reward -= min(bumpingCostFactor*fClassPrices[nFareClasses-1],maxBumpingCost)*nPaxToBump # DBC - Denied Boarding Compensation
                                bumpingCost = min(bumpingCostFactor*fClassPrices[nFareClasses-1],maxBumpingCost)*nPaxToBump
                            elif((nPaxToBump-nBookings_fClass[nFareClasses-1]) <= nBookings_fClass[nFareClasses-2]):
                                # first bump all pax of lowest fare class
                                reward -= min(bumpingCostFactor*fClassPrices[nFareClasses-1],maxBumpingCost)*nBookings_fClass[nFareClasses-1]
                                bumpingCost = min(bumpingCostFactor*fClassPrices[nFareClasses-1],maxBumpingCost)*nBookings_fClass[nFareClasses-1]
                                nPaxToBump -= nBookings_fClass[nFareClasses-1]                                        
                                nBookings_fClass[nFareClasses-1] = 0                    
                                # then bump middle class pax until nPaxOnBoard = capacity
                                reward -= min(bumpingCostFactor*fClassPrices[nFareClasses-2],maxBumpingCost)*nPaxToBump
                                bumpingCost += min(bumpingCostFactor*fClassPrices[nFareClasses-2],maxBumpingCost)*nPaxToBump 
                                nBookings_fClass[nFareClasses-2] -= nPaxToBump
                                # nPaxToBump = 0
                            else: # elif((nPaxToBump-nBookings_fClass[nFareClasses-1]-nBookings_fClass[nFareClasses-2]) <= nBookings_fClass[nFareClasses-3]):
                                # first bump all pax of lowest fare class 
                                reward -= min(bumpingCostFactor*fClassPrices[nFareClasses-1],maxBumpingCost)*nBookings_fClass[nFareClasses-1]
                                nPaxToBump -= nBookings_fClass[nFareClasses-1]
                                nBookings_fClass[nFareClasses-1] = 0
                                # then bump all middle class pax
                                reward -= min(bumpingCostFactor*fClassPrices[nFareClasses-2],maxBumpingCost)*nBookings_fClass[nFareClasses-2]
                                nPaxToBump -= nBookings_fClass[nFareClasses-2]
                                nBookings_fClass[nFareClasses-2] = 0
                                # then bump high fare class pax until nPaxToBump=0
                                reward -= min(bumpingCostFactor*fClassPrices[nFareClasses-3],maxBumpingCost)*nPaxToBump
                                # nPaxToBump = 0
                                nBookings_fClass[nFareClasses-3] -= nPaxToBump
            

                    observation = (time+1, list(nBookings_fClass), list(nCancellations))
                    
                    
                    nSeatsAvailable = capacity - sum(nBookings_fClass) # update nSeatsAvailable
                            


                    episode_reward += reward
                    
                    # print('testLogger.observations[currentEpisodeIndex] =',testLogger.observations[currentEpisodeIndex],'\n')
                    testLogger.on_step_end(currentEpisodeIndex, observation, reward, fClassOpen, BLemsrb)
                    
                testLogger.on_episode_end(nBookings_fClass, nCancellations, episode_reward, max_reward, overbooking, capacity) 
                
                
            

            ##t1 = np.arange(nTestEpisodes)+1
            ##t2 = np.arange(1+(rmInterval-1)/2, nTestEpisodes+1-(rmInterval-1)/2)
            rP = np.array(testLogger.rewardPercentage)*100
            # aP = np.array(test_logger.acceptPercentage)*100
            LF = np.array(testLogger.loadFactor)*100
            # np.savez(outputDir + '/TrainingData.npz', rP=rP, aP=aP, LF=LF)

            print('avgOptimalRewardPercentage = ' + repr(np.mean(rP)))
            # print('average_ap = ' + repr(np.mean(aP)))
            print('avgLoadFactor = ' + repr(np.mean(LF)))

            pklFilename = "emsrbAgent_testLogger_cnc" + repr(cancelProbsSetting) + "_fcArrivals" + repr(meanNarrivalsSetting) + ".pkl"            
            with open(pklFilename, 'wb') as pklFilePointer:
                pickle.dump([testLogger.nBookings_fClass, testLogger.nCancellations, testLogger.episode_reward,
                             testLogger.max_reward, testLogger.rewardPercentage, testLogger.overbooking,
                             testLogger.loadFactor, testLogger.observations, testLogger.rewards, testLogger.fClassOpen, testLogger.BLemsrb], pklFilePointer)

            f.write(repr(bumpingCostFactor) + ", ")
            f.write(repr(cancelProbsSetting) + ", ")
            f.write(repr(meanNarrivalsSetting) + ", ")
            f.write(repr(np.mean(rP)) + ", ")
            f.write(repr(np.mean(LF)) + "\n")

f.close()
