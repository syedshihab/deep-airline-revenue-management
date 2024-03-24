# TODO: change armClasses...py name if necessary, trainingData filename (before training and before testing), nbActions, nerual network input, nHiddenNeurons, nTestEpisodes if necessary after
# changing market parameters
# TODO: change nTrEpisodes and indexSimulationRun before each new run
# TODO: change the filename (modelIndex and simulationRunIndex) of the best model before loading weights onto the neural network

# list of hyperparameters: target_model_update, nTrEpisodes, gamma, nHiddenNeuronsEachLayer, nHiddenLayers, activation function, lr, policy

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
from armClasses_observationSpaceModified1_stateRewardScaled_2 import *

biased = True # bias is present
computeRewardAtEnd = False # (immediate) reward is calculated after each action
bumpingCostFactor = 2

#############################################################################################################################################################################
target_model_update = 10000 # 1e-3 # 1e-2 CHANGED # 10000 # 1e-2 # 0.01  -> 0.01<1, so soft update of target model: target_model_params = target_model_update * local_model_params + (1 - target_model_update) * taget_model_params
# TODO: try target_model_update = 10,000 (hard update every 10000th step)
#############################################################################################################################################################################


#############################################################################################################################################################################

indexSimulationRun = 1005 # 1004 # 1003 # 1002 # the index simulation run associated with the model being tested

#############################################################################################################################################################################

gamma = 0.9999
test_policy = GreedyQPolicy() # default testPolicy
arm_processor = armProcessor(biased)
nb_actions = 6 # 18 # 6 # 18
nStateVars = 1+2+2+2 


#############################################################################################################################################################################

nHiddenNeuronsByLayer = [256, 256, 256] # [1024, 1024, 1024] # [16, 16, 16] # [512, 512, 512, 64] # [256, 256, 128, 64] # [512, 512, 512, 64] # [256, 256, 128, 64] # nHiddenNeuronsByLayer = [512] # nHiddenNeuronsByLayer = [512, 512]
##nHiddenNeuronsLayer1 = 256
##nHiddenNeuronsLayer2 = 256
##nHiddenNeuronsLayer3 = 128
##nHiddenNeuronsLayer4 = 64
nHiddenNeurons = 256
nHiddenLayers = 3 # 2

#############################################################################################################################################################################


# Build model TODO: neural network hyperparameter optimization
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
model.add(Dense(nb_actions))
model.add(Activation('linear'))

#############################################################################################################################################################################

memory = SequentialMemory(limit=50000, window_length=1)


### during "training" or "market adaption" phase, agent should be allowed to explore to a limited degree to learn faster
##
##nb_max_start_steps = 100 # Number of maximum steps that the agent performs at the beginning of each episode using `start_step_policy`. Notice that this is an upper limit since the exact number of steps...
##                         # to be performed is sampled uniformly from [0, max_start_steps] at the beginning of each episode.
##
### after 100 days, the number of arrivals of high fare product start increasing rapidly
##start_step_policy = EpsGreedyQPolicy(eps=.1)
### after t=100, there are two optiions for exploration policy
##policy = EpsGreedyQPolicy(eps=.05)
##policy = GreedyQPolicy()
##
### during testing, exploration should be zero to get the best result/performance
##policy = GreedyQPolicy()

#############################################################################################################################################################################
# during testing, exploration should be zero to get the best result/performance
value_max=0.1
value_min=0
nEps_expl = 100 # TODO: vary
nStepsPerEps = 182
# policy: decrease epsilon from 0.1 to 0 over the 'nEps_expl' episodes and then follow a greedy policy
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=value_max, value_min=value_min, value_test=.05, nb_steps=nEps_expl*nStepsPerEps) # nb_steps=nb_steps-nStepsPerEps*500
# policy = GreedyQPolicy()
#############################################################################################################################################################################

dqn = DQNAgent(processor=arm_processor, model=model, test_policy=test_policy, nb_actions=nb_actions,
               memory=memory, policy=policy, target_model_update=target_model_update, gamma=gamma) # gamma = discount factor

#############################################################################################################################################################################
# TODO: CHANGED LEARNING RATE
dqn.compile(Adam(lr=1e-4), metrics=['mae']) # lr=.00025; lr = 1e-2; lr = 1e-3 lr=1e-5
#############################################################################################################################################################################


#############################################################################################################################################################################
# loading weights of best model of the best DRL agent (the best NN architecture and hyperparameters)
outputDir = "output_OB2cnc1fd1_2fp22pp_twg1wg4c1c2_nHL3nHN256tmu1lr1pol123_bestAgent"
bestModelIndex = 32 # 52 # 35
dqn.load_weights(outputDir + '/dqn_armEnv_weights_model{}.h5f'.format(bestModelIndex+1)) 
#############################################################################################################################################################################

test_logger = infoLogger()

# TODO: CHANGED nTestEpisodes
nTestEpisodes = 2000 # 4000 # 2000 # 5000 # 2000

logging_interval = nTestEpisodes/500

testData = "testData_cnc1_fcArrivals1_2fp22pp_true_mktSiml" + ".mat"

env = armEnv(testData, biased, computeRewardAtEnd, bumpingCostFactor) # creating armEnv instance

history = dqn.fit(env, callbacks=[test_logger], verbose=0, nb_steps=nTestEpisodes*nStepsPerEps, log_interval=logging_interval) 
# saving the current, more experienced agent (NN model)
dqn.save_weights(outputDir + '/dqn_armEnv_weights_bestModel_2000exp_trueMktSiml.h5f', overwrite=True)

RP = np.array(test_logger.rewardPercentage)*100
LF = np.array(test_logger.loadFactor)*100

print('avgOptimalRewardPercentage of best model in the last 300 episodes = ' + repr(np.mean(RP[-300:])))
print('avgLoadFactor of best model in the last 300 episodes = ' + repr(np.mean(LF[-300:])))

# saving the test_logger object
with open(outputDir + '/testLogger_bestModel_trueMktSiml_lrNeg4_tmu10k.pkl', 'wb') as pklFilePointer1:
    pickle.dump([test_logger.nBookings_fp, test_logger.nBookings_price, test_logger.preBumping_nBookings_fp, test_logger.preBumping_nBookings_price, test_logger.bumpingCost,
                 test_logger.nCancellations_price, test_logger.episode_reward, test_logger.max_reward, test_logger.rewardPercentage, test_logger.overbooking,
                 test_logger.loadFactor, test_logger.observations, test_logger.rewards, test_logger.actions], pklFilePointer1)


# Plotting test_logger.rewardPercentage
t1 = np.arange(nTestEpisodes)+1

# np.savez(outputDir + '/TrainingData.npz', RP=RP, aP=aP, LF=LF)

rmInterval = 111 # 55
t2 = np.arange(1+(rmInterval-1)/2, nTestEpisodes+1-(rmInterval-1)/2)
plt.plot(t2, running_mean(RP, rmInterval))
# axes = plt.gca()
# axes.set_xlim([1,nTestEpisodes])
# axes.set_ylim([70,100]) # TODO: CHANGED ([0,100])
# plt.show()
# plt.title('Learning Curve')
plt.ylabel('Optimal Revenue (%)')
plt.xlabel('Episode')
##plt.show()
plt.savefig(outputDir + '/testPlot_bestModel_trueMktSiml_optimalRP.pdf')
plt.close()
# plt.show()

##rmInterval = 111 # 55
##t2 = np.arange(1+(rmInterval-1)/2, nTestEpisodes+1-(rmInterval-1)/2)
plt.plot(t2, running_mean(LF, rmInterval))
# axes = plt.gca()
# axes.set_xlim([1,nTestEpisodes])
# axes.set_ylim([70,100]) # TODO: CHANGED ([0,100])
# plt.show()
# plt.title('Learning Curve')
plt.ylabel('Load factor (%)')
plt.xlabel('Episode')
##plt.show()
plt.savefig(outputDir + '/testPlot_bestModel_trueMktSiml_LF.pdf')
plt.close()

