# This scripts generates agent's perfomance metrics during testing
# In this script, both the DRL agent and the EMSRb agent are tested on the same set of test episodes (nTestEpisodes = 300)

# TODO: add variance, confidence interval

import numpy as np
import gym
from gym import spaces
import scipy.io
import pdb
import time
import pickle

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.utils import plot_model

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor, Env

import matplotlib.pyplot as plt
import os

# from armClasses_dynamicPricing import *
# from armClasses_observationSpaceModified1 import *
# from armClasses_observationSpaceModified1_stateRewardScaled_SIC import *
from DRLagent_armClasses_3fClasses_Tb1b2b3_SIC import *

biased = True # bias is present
computeRewardAtEnd = False # (immediate) reward is calculated after each action

target_model_update = 1e-2 # 10000 # 1e-2 # 0.01  -> 0.01<1, so soft update of target model: target_model_params = target_model_update * local_model_params + (1 - target_model_update) * taget_model_params
# TODO: try target_model_update = 10,000 (hard update every 10000th step)

nTrEpisodes = 25000 # it has to be some multiple of checkPointInterval # 50000 # 1000 # 50000 # 100000 # 20000 # 99000 # 10000 # 10000 # 49000 # 10000 # 20000
indexSimulationRun = 1

gamma = 0.995
nb_eps_steps = 182
nb_steps = nb_eps_steps*nTrEpisodes  # 10,000 episodes; TODO: vary, try 182x20000 -> 20000 episodes; number of steps/state transitions in each episode = bookingHorizon = 182
logging_interval = nb_steps/10 # TODO: check convergence criteria/method/mechanism/strategy
rmInterval = 111 # 333 # 11 # 111 # 333 # 555 # choose odd running mean interval # int(nb_steps/1000) # TODO: use replay buffer
# testInterval = 50

value_max=1
value_min=.01
# policyStr = "LinearAnnealedPolicy(BoltzmannQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=nb_steps)"
policyStr = "LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.01, value_test=.05, nb_steps=nb_steps)"                              
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.01, value_test=.05, nb_steps=nb_steps) # nb_steps=nb_steps-nb_eps_steps*500
# policy = EpsGreedyQPolicy(eps=.1)
# policy = LinearAnnealedPolicy(BoltzmannQPolicy(), attr='tau', value_max=1., value_min=.1, value_test=.05, nb_steps=nb_steps)
test_policy = GreedyQPolicy() # default testPolicy

arm_processor = armProcessor(biased)

nActions = 3 # 6 # 18 # 6 # 18

nHiddenNeuronsByLayer = [256, 256, 256, 64] # [256, 256] # [1024, 1024, 1024] # [16, 16, 16] # [512, 512, 512, 64] # [256, 256, 128, 64] # [512, 512, 512, 64] # [256, 256, 128, 64] # nHiddenNeuronsByLayer = [512] # nHiddenNeuronsByLayer = [512, 512]
##nHiddenNeuronsLayer1 = 256
##nHiddenNeuronsLayer2 = 256
##nHiddenNeuronsLayer3 = 128
##nHiddenNeuronsLayer4 = 64
nHiddenNeurons = 256
nHiddenLayers = 3

##activationFunc = 'relu'

# Build model TODO: neural network hyperparameter optimization
nStateVars = 4 # (t, b1, b2, b3)
model = Sequential()
if(biased):
    model.add(Flatten(input_shape=(1,nStateVars+1))) # 8 # 12 # input to NN: ([time, seats[0], seats[1], seats[2], bias=1]) array
else:
    model.add(Flatten(input_shape=(1,nStateVars))) # 7 # 11
model.add(Dense(nHiddenNeuronsByLayer[0])) # 128 # 16, 48 gives average_rp ~ 80% only # (16,16,16), (256,256), (256,256,128)
model.add(Activation('relu'))
model.add(Dense(nHiddenNeuronsByLayer[1])) # 128
model.add(Activation('relu'))
model.add(Dense(nHiddenNeuronsByLayer[2]))
model.add(Activation('relu'))
##model.add(Dense(nHiddenNeuronsByLayer[3]))
##model.add(Activation('relu'))
model.add(Dense(nActions))
model.add(Activation('linear'))


memory = SequentialMemory(limit=50000, window_length=1)




# logging experimental results
performanceLogDir = "numericalExperimentsResults_500" # location where results.txt file (containing performance metrics values of each simulation run) is saved
if not os.path.isdir(performanceLogDir): 
    os.mkdir(performanceLogDir)

filePointer1 = open(performanceLogDir+"/results_500.txt", "w")
filePointer1.write("bumpingCostFactor, cancelProbsSetting, meanNarrivalsSetting, bestModelIndex, test_mean RP of best model, test_mean LF of best model" + "\n")


# Running numerical experiments for sensitivity analysis
bcfArray = [2] # [2.0, 2.5]
cncArray = [1,2,3] # [1] # [ 2 3]
fdArray = [1,2,3] # [1 2 3] # fdArray = [3] # changed from fdArray = [1] 
for bumpingCostFactor in bcfArray:
    for cancelProbsSetting in cncArray:
        for meanNarrivalsSetting in fdArray:

            
            outputDir = "numExp_outputs_" + "OB" + repr(bumpingCostFactor) + "cnc" + repr(cancelProbsSetting) + "_fcArrivals" + repr(meanNarrivalsSetting) + "_" + repr(indexSimulationRun) # location where output plots and model weights are saved # TODO: change last digit
            

            # Creating DQN agent
            dqn = DQNAgent(processor=arm_processor, model=model, test_policy=test_policy, nb_actions=nActions,
                           memory=memory, policy=policy, target_model_update=target_model_update, gamma=gamma) # gamma = discount factor

            dqn.compile(Adam(lr=1e-3), metrics=['mae']) # lr=.00025


            # loading trModels_testPerf_meanRP
            with open(outputDir + '/trRP_trLF_modelsTestRP_modelsTestLF.pkl','rb') as filePointer0:
                tr_RP, tr_LF, trModels_testPerf_meanRP, trModels_testPerf_meanLF = pickle.load(filePointer0)

            

            bestModelIndex = trModels_testPerf_meanRP.index(max(trModels_testPerf_meanRP))
            print('Best model = ' + repr(bestModelIndex+1))
            
            

            
            # loading best model and then testing its performance and saving its testLogger object
            dqn.load_weights(outputDir + '/dqn_armEnv_weights_model{}.h5f'.format(bestModelIndex+1)) # model358
                        
            testData = "testData_cnc" + repr(cancelProbsSetting) + "_fcArrivals" + repr(meanNarrivalsSetting) + ".mat"  # same as the one for the EMSRb agent
            env = armEnv(testData, biased, computeRewardAtEnd, bumpingCostFactor) # creating armEnv instance

            test_logger = infoLogger()
            nTestEpisodes = 300 # same as the one for the EMSRb agent
            
                       
            dqn.test(env, callbacks = [test_logger], nb_episodes=nTestEpisodes, visualize=False)

            
            rP = np.array(test_logger.rewardPercentage)*100
            # aP = np.array(test_logger.acceptPercentage)*100
            LF = np.array(test_logger.loadFactor)*100
            # np.savez(outputDir + '/TrainingData.npz', rP=rP, aP=aP, LF=LF)

            print('avgOptimalRewardPercentage of best model = ' + repr(np.mean(rP)))
            # print('average_ap = ' + repr(np.mean(aP)))
            print('avgLoadFactor of best model = ' + repr(np.mean(LF)))

            # saving the test_logger object of the best model
            with open(outputDir + '/testLogger_bestModel_500.pkl', 'wb') as pklFilePointer1:
                pickle.dump([test_logger.nBookings, test_logger.nCancellations, test_logger.episode_reward,
                             test_logger.max_reward, test_logger.rewardPercentage, test_logger.overbooking,
                             test_logger.loadFactor, test_logger.observations, test_logger.rewards, test_logger.actions], pklFilePointer1)   

            

            # writing simulation/experiment details onto files problemParameters.txt and results.txt
            filePointer1.write(repr(bumpingCostFactor) + ", ")
            filePointer1.write(repr(cancelProbsSetting) + ", ")
            filePointer1.write(repr(meanNarrivalsSetting) + ", ")
            filePointer1.write(repr(bestModelIndex+1) + ", ") # starting index = 1
            filePointer1.write(repr(np.mean(rP)) + ", ")
            filePointer1.write(repr(np.mean(LF)) + "\n")

            # TODO: add variance
            


filePointer1.close()



