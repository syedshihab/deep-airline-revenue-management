import numpy as np
import gym
from gym import spaces
import scipy.io
import pdb
import time 

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

from armClasses_observationSpaceModified1 import *

biased = True # bias is present
computeRewardAtEnd = False # (immediate) reward is calculated after each action
target_model_update = 10000 # 1e-2 # 0.01  -> 0.01<1, so soft update of target model: target_model_params = target_model_update * local_model_params + (1 - target_model_update) * taget_model_params
# TODO: try target_model_update = 10,000 (hard update every 10000th step)
nEpisodes = 1000 # 49000 # 10000 # 20000
nb_steps = 182*nEpisodes  # 10,000 episodes; TODO: vary, try 182x20000 -> 20000 episodes; number of steps/state transitions in each episode = bookingHorizon = 182
logging_interval = nb_steps/10 # TODO: check convergence criteria/method/mechanism/strategy
rmInterval = 111 # 555 # choose odd running mean interval # int(nb_steps/1000) # TODO: use replay buffer
# testInterval = 50
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=nb_steps)
# policy = EpsGreedyQPolicy(eps=.1)
# policy = LinearAnnealedPolicy(BoltzmannQPolicy(), attr='tau', value_max=1., value_min=.1, value_test=.05,
                              # nb_steps=nb_steps)
# test_policy = GreedyQPolicy()

arm_processor = armProcessor(biased)
nb_actions = 18

# Build model TODO: neural network hyperparameter optimization
model = Sequential()
if(biased):
    model.add(Flatten(input_shape=(1,12))) # input to NN: ([time, seats[0], seats[1], seats[2], bias=1]) array
else:
    model.add(Flatten(input_shape=(1,11)))
model.add(Dense(256)) # 128 # 16, 48 gives average_rp ~ 80% only
model.add(Activation('relu'))
model.add(Dense(256)) # 128
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

test_policy = EpsGreedyQPolicy(eps=1)
# dqn.test_policy = EpsGreedyQPolicy(eps=1)
memory = SequentialMemory(limit=50000, window_length=1)

dqn = DQNAgent(processor=arm_processor, model=model, test_policy=test_policy, nb_actions=nb_actions,
        memory=memory, policy=policy, 
        target_model_update=target_model_update, gamma=0.995) # gamma = discount factor

dqn.compile(Adam(lr=1e-3), metrics=['mae']) # lr=.00025

# Running numerical experiments for sensitivity analysis
bcfArray = [2] # [2.0, 2.5]
cncArray = [1] # [ 2 3]
fdArray = [1] # [1 2 3] # fdArray = [3] # changed from fdArray = [1]
for bumpingCostFactor in bcfArray:
    for cnc in cncArray:
        for fd in fdArray:
            
            trainingData = "training_c" + repr(cnc) + "_fd" + repr(fd) + "_testDP5" + ".mat"  # 'training_c1_fd1_testDP5'
            env = armEnv(trainingData, biased, computeRewardAtEnd, bumpingCostFactor) # creating armEnv instance

            randomAgent_logger = infoLogger() 
            nTestEpisodes = 1000            
            dqn.test(env, callbacks = [randomAgent_logger], nb_episodes=nTestEpisodes, verbose=0, visualize=False)
         
            rP = np.array(randomAgent_logger.rewardPercentage)*100
            # aP = np.array(randomAgent_logger.acceptPercentage)*100
            LF = np.array(randomAgent_logger.loadFactor)*100
            # np.savez(outputDir + '/TrainingData.npz', rP=rP, aP=aP, LF=LF)

            print('avgOptimalRewardPercentage = ' + repr(np.mean(rP)))
            # print('average_ap = ' + repr(np.mean(aP)))
            print('avgLoadFactor = ' + repr(np.mean(LF)))



            

            
