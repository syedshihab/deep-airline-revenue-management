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
from armClasses_observationSpaceModified1_stateRewardScaled import *

biased = True # bias is present
computeRewardAtEnd = False # (immediate) reward is calculated after each action

target_model_update = 1e-2 # 10000 # 1e-2 # 0.01  -> 0.01<1, so soft update of target model: target_model_params = target_model_update * local_model_params + (1 - target_model_update) * taget_model_params
# TODO: try target_model_update = 10,000 (hard update every 10000th step)

nTrEpisodes = 100000 # 20000 # 99000 # 10000 # 10000 # 49000 # 10000 # 20000
indexSimulationRun = 555


gamma = 0.995
nb_eps_steps = 182
nb_steps = nb_eps_steps*nTrEpisodes  # 10,000 episodes; TODO: vary, try 182x20000 -> 20000 episodes; number of steps/state transitions in each episode = bookingHorizon = 182
logging_interval = nb_steps/10 # TODO: check convergence criteria/method/mechanism/strategy
rmInterval = 333 # 11 # 111 # 333 # 555 # choose odd running mean interval # int(nb_steps/1000) # TODO: use replay buffer
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

nb_actions = 6 # 18 # 6 # 18

nHiddenNeuronsByLayer = [256, 256, 256, 64] # [1024, 1024, 1024] # [16, 16, 16] # [512, 512, 512, 64] # [256, 256, 128, 64] # [512, 512, 512, 64] # [256, 256, 128, 64] # nHiddenNeuronsByLayer = [512] # nHiddenNeuronsByLayer = [512, 512]
##nHiddenNeuronsLayer1 = 256
##nHiddenNeuronsLayer2 = 256
##nHiddenNeuronsLayer3 = 128
##nHiddenNeuronsLayer4 = 64
nHiddenNeurons = 256
nHiddenLayers = 3

##activationFunc = 'relu'

# Build model TODO: neural network hyperparameter optimization
model = Sequential()
if(biased):
    model.add(Flatten(input_shape=(1,8))) # 8 # 12 # input to NN: ([time, seats[0], seats[1], seats[2], bias=1]) array
else:
    model.add(Flatten(input_shape=(1,7))) # 7 # 11
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


memory = SequentialMemory(limit=50000, window_length=1)

dqn = DQNAgent(processor=arm_processor, model=model, test_policy=test_policy, nb_actions=nb_actions,
        memory=memory, policy=policy, target_model_update=target_model_update, gamma=gamma) # gamma = discount factor

dqn.compile(Adam(lr=1e-3), metrics=['mae']) # lr=.00025

######## Specify Output directory and model number #################
# loading weights of the best model for trainingData = training_c1_fd1_startTr5 when cancellationRate = [0, 0]
dqn.load_weights('output_OB2cnc1fd1_53' + '/dqn_armEnv_weights_model358.h5f')

# bcfArray = [2.5]

# logging experimental results
performanceLogDir = "numericalExperimentsResults" # location where results.txt file (containing performance metrics values of each simulation run) is saved
if not os.path.isdir(performanceLogDir): 
    os.mkdir(performanceLogDir)

f = open(performanceLogDir+"/results.txt", "w")
f.write("wtpGroupMean, cncSetting, bumpingCostFactor" + "\n")


# Running numerical experiments for sensitivity analysis
bcfArray = [2] # [2.0, 2.5]
cncArray = [1] # [ 2 3]
fdArray = [1] # [1 2 3] # fdArray = [3] # changed from fdArray = [1] 
for bumpingCostFactor in bcfArray:
    for cnc in cncArray:
        for fd in fdArray:

            
            outputDir = "output_" + "OB" + repr(bumpingCostFactor) + "cnc" + repr(cnc) + "fd" + repr(fd) + "_" + repr(indexSimulationRun) # location where output plots and model weights are saved # TODO: change last digit
            if not os.path.isdir(outputDir):
                os.mkdir(outputDir)
                   
            trainingData = "training_c" + repr(cnc) + "_fd" + repr(fd) + "_startTr7" + ".mat"  # 'training_c1_fd1_startTr' # 'training_c1_fd1_testDP6' # 'training_c1_fd1_testDP5.mat'

            env = armEnv(trainingData, biased, computeRewardAtEnd, bumpingCostFactor) # creating armEnv instance

            info_logger = infoLogger()
            test_logger = infoLogger()

##            ModelIntervalCheckpoint_logger = ModelIntervalCheckpoint(filepath=outputDir + '/dqn_armEnv_weights_trEps{}.h5f', interval=182*500, verbose=0) # save model weights after every 500 episodes
##            # dqn.test(env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True, nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1)
##            
##            start = time.time()
##            print('Starting training')
##            history = dqn.fit(env, callbacks=[info_logger, ModelIntervalCheckpoint_logger], verbose=0, nb_steps=nb_steps, log_interval=logging_interval) # do: try verbose=1, visualize = True; TODO: why env is not rendering?
##            print('Completed training')
##            # history = a keras.callbacks.History instance that recorded the entire training process.
##            end = time.time()
##            print('training time =',(end-start)/3600, 'hrs')
##            
##            dqn.save_weights(outputDir + '/dqn_armEnv_weights.h5f', overwrite=True)
            # TODO: check if NN weight updated during training as per target_model_update rule? Updated NN used in dqn.test() when it is called later on?
            # TODO: save dqn models at different intervals of the training, test its performance -> to find the best dqn model


            # save weights and test model after every 500 episodes
            nTestEpisodes = 100
            stepsCounter = 0
            tr_RP = [] # records RP of agent in all training episodes
            tr_LF = [] # records LF of agent in all training episodes
            trModels_meanRP = [] # records mean RP (test performance) of agent in test episodes at all the checkpoints 
            trModels_meanLF = [] # records mean LF (test performance) of agent in test episodes at all the checkpoints 
            checkPointInterval = 250
            while (stepsCounter<nb_steps):
                
                
                # Run training for 250 episodes (= 250*182 steps)
                history = dqn.fit(env, callbacks=[info_logger], verbose=0, nb_steps=checkPointInterval*182, log_interval=logging_interval) 

                stepsCounter += checkPointInterval*182
                value_max = max(value_min, 1-((1-value_min)/nb_steps)*stepsCounter) # value_max = policy.get_current_value()
                dqn.policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=value_max, value_min=.01, value_test=.05, nb_steps=(nb_steps-stepsCounter)) # nb_steps=nb_steps-nb_eps_steps*500
                
                tr_RP.append(info_logger.rewardPercentage)
                tr_LF.append(info_logger.loadFactor)
                
##                # Save neural network weights
##                dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
##
##                # Save memory
##                pickle.dump(memory, open("memory.pkl", "wb"))

                # Run test
                dqn.test(env, callbacks = [test_logger], nb_episodes=nTestEpisodes, visualize=False)
                iterInd = int((stepsCounter/(checkPointInterval*182)))                
                # rP = np.array(test_logger.rewardPercentage[iterInd*300-300:iterInd*300])*100
                rP = np.array(test_logger.rewardPercentage)*100
                trModels_meanRP.append(np.mean(rP))                
                # LF = np.array(test_logger.loadFactor[iterInd*300-300:iterInd*300])*100
                LF = np.array(test_logger.loadFactor)*100
                trModels_meanLF.append(np.mean(LF))
                print('mean RP of current model =', np.mean(rP))
                print('mean LF of current model =', np.mean(LF))

                print('current episode index =', iterInd*checkPointInterval)
                print('epsilon value_max =', value_max)

                dqn.save_weights(outputDir + '/dqn_armEnv_weights_model{}.h5f'.format(iterInd), overwrite=True)
                # dqn.load_weights(outputDir + '/dqn_armEnv_weights_model{}.h5f'.format(iterInd))
                # weights = dqn.model.get_weights()

            bestModelIndex = trModels_meanRP.index(max(trModels_meanRP))
            print('Best model = ' + repr(bestModelIndex+1))
            print('mean RP of best model = ' + repr(max(trModels_meanRP)))
            print('mean LF of best model = ' + repr(trModels_meanLF[bestModelIndex]))
            
            # Making plots of rP of trModels at the checkpoints 
            plt.plot(trModels_meanRP)
            # plt.title('Learning Curve')
            plt.ylabel('Percentage of Optimal Reward')
            plt.xlabel('Model')
            plt.savefig(outputDir + '/trainingPlot_optimalRewardPercentage.png')
            plt.close()
            # plt.show()

            # Making plots of rP of trModels at the checkpoints 
            plt.plot(trModels_meanLF)
            # plt.title('Learning Curve')
            plt.ylabel('Percentage of Load Factor')
            plt.xlabel('Model')
            plt.savefig(outputDir + '/trainingPlot_loadFactorPercentage.png')
            plt.close()
            # plt.show()
                


            # saving the testLogger object created by the best model
            # specify best model
            dqn.load_weights(outputDir + '/dqn_armEnv_weights_model{}.h5f'.format(bestModelIndex+1)) # model358
            

            testData = "training_c" + repr(cnc) + "_fd" + repr(fd) + "_startTr6_test" + ".mat"  # 'training_c1_fd1_startTr' # 'training_c1_fd1_testDP6' # 'training_c1_fd1_testDP5.mat'
            env = armEnv(testData, biased, computeRewardAtEnd, bumpingCostFactor) # creating armEnv instance

            test_logger = infoLogger()
            nTestEpisodes = 300
            rmInterval = int(nTestEpisodes/10)

            # TODO: use dqn.test_policy or dqn.fit(exploration=0 i.e. policy=greedyQplolicy) check if policy=greedyQplolicy in dqn.test()
            # env.currentEpisodeIndex = nTrEpisodes
            dqn.test(env, callbacks = [test_logger], nb_episodes=nTestEpisodes, visualize=False)

            # nTestEpisodes = len(rP)
            t1 = np.arange(nTestEpisodes)+1
            t2 = np.arange(1+(rmInterval-1)/2, nTestEpisodes+1-(rmInterval-1)/2)
            rP = np.array(test_logger.rewardPercentage)*100
            # aP = np.array(test_logger.acceptPercentage)*100
            LF = np.array(test_logger.loadFactor)*100
            # np.savez(outputDir + '/TrainingData.npz', rP=rP, aP=aP, LF=LF)

            print('avgOptimalRewardPercentage = ' + repr(np.mean(rP)))
            # print('average_ap = ' + repr(np.mean(aP)))
            print('avgLoadFactor = ' + repr(np.mean(LF)))

            # saving the test_logger object
            with open(outputDir + '/testLogger.pkl', 'wb') as pklFilePointer:
                pickle.dump([test_logger.seats, test_logger.nSeatsSoldByPrices, test_logger.nCancellationsByPrices, test_logger.episode_reward,
                             test_logger.max_reward, test_logger.rewardPercentage, test_logger.overbooking,
                             test_logger.loadFactor, test_logger.observations, test_logger.rewards, test_logger.actions], pklFilePointer)   

            # saving tr_RP, tr_LF, trModels_meanRP and trModels_meanLF
            with open(outputDir + '/trRP_trLF_modelsRP_modelsLF.pkl', 'wb') as pklFilePointer2:
                pickle.dump([tr_RP, tr_LF, trModels_meanRP, trModels_meanLF], pklFilePointer2)            
            
##            # TODO: dqn.load_weights() (and also load memory using pickle if you are creating a new agent using dqn=dqnAgent(...)
##            # two stage annealing process
##            dqn.policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.01, value_min=.001, value_test=.05, nb_steps=nb_steps) # nb_steps=nb_steps-nb_eps_steps*500
##            history = dqn.fit(env, callbacks=[info_logger], verbose=0, nb_steps=500, log_interval=logging_interval) 



            
            
##            rP = np.array(info_logger.rewardPercentage)*100
##            nTrEpisodes = len(rP)
##            t1 = np.arange(nTrEpisodes)+1 # 1,2,...,nTrEpisodes
##            t2 = np.arange(1+(rmInterval-1)/2, nTrEpisodes+1-(rmInterval-1)/2) # for odd rmInterval # t2 = np.arange(rmInterval/2-1, nTrEpisodes-rmInterval/2) 
##            # aP = np.array(info_logger.acceptPercentage)*100
##            LF = np.array(info_logger.loadFactor)*100
##
##            # saving rP and LF arrays
##            np.savez(outputDir + '/TrainingData.npz', rP=rP, LF=LF)
##            
##
##            # Making training plots of rP, LF and fare class seat allocations vs episodeNumber
##            # pdb.set_trace()
##            plt.plot(t1, rP, t2, running_mean(rP, rmInterval), 'r')
##            # plt.title('Learning Curve')
##            plt.ylabel('Percentage of Optimal Reward')
##            plt.xlabel('Episode')
##            plt.savefig(outputDir + '/trainingPlot_optimalRewardPercentage.png')
##            plt.close()
##            # plt.show()
##
##
##            plt.plot(t1, LF, t2, running_mean(LF, rmInterval), 'r')
##            # plt.title('Load factor')
##            plt.ylabel('Load factor')
##            plt.xlabel('Episode')
##            plt.savefig(outputDir + '/trainingPlot_loadFactor.png')
##            plt.close()
##
##            plt.plot(info_logger.seats)
##            plt.title('Seat Allocation')
##            plt.ylabel('Number of Seats per Class')
##            plt.xlabel('Episode')
##            plt.savefig(outputDir + '/trainingPlot_fclassSeatAllocation.png')
##            # plt.show()
##            plt.close()
####
####            # After training is done, we save the final weights.
####            dqn.save_weights('dqn_weights.h5f', overwrite=True)
##
##            # plot_model(model, to_file='model.png', show_shapes=True)
##
##            print('Performance of the agent in the last 100 episodes')
##            print('avgOptimalRewardPercentage = ' + repr(np.mean(rP[-100:])))
##            # print('average_ap = ' + repr(np.mean(aP)))
##            print('avgLoadFactor = ' + repr(np.mean(LF[-100:])))

            

            # TODO: why optimalRewardPercentage = 0 in some episodes? Check trainingData, agent actions in those episodes.
            # dqn.test calls env.reset(). So the policy is tested on the first nb_episodes of trainingData


##            test_logger = infoLogger()
##            nTestEpisodes = 300
##            rmInterval = int(nTestEpisodes/10)
##
##            # TODO: use dqn.test_policy or dqn.fit(exploration=0 i.e. policy=greedyQplolicy) check if policy=greedyQplolicy in dqn.test()
##            # env.currentEpisodeIndex = nTrEpisodes
##            dqn.test(env, callbacks = [test_logger], nb_episodes=nTestEpisodes, visualize=False)
##
##            # nTestEpisodes = len(rP)
##            t1 = np.arange(nTestEpisodes)+1
##            t2 = np.arange(1+(rmInterval-1)/2, nTestEpisodes+1-(rmInterval-1)/2)
##
##            rP = np.array(test_logger.rewardPercentage)*100
##            # aP = np.array(test_logger.acceptPercentage)*100
##            LF = np.array(test_logger.loadFactor)*100
##            # np.savez(outputDir + '/TrainingData.npz', rP=rP, aP=aP, LF=LF)
##
##            print('avgOptimalRewardPercentage = ' + repr(np.mean(rP)))
##            # print('average_ap = ' + repr(np.mean(aP)))
##            print('avgLoadFactor = ' + repr(np.mean(LF)))
##
##            # Interested in the performance of the agent on TEST DATA
##            
##            # looking at the 300 episodes of training

            # writing simulation/experiment details onto files problemParameters.txt and results.txt
            f.write(repr(fd) + ", ")
            f.write(repr(cnc) + ", ")
            f.write(repr(bumpingCostFactor) + ", ")
            
            g = open(outputDir+"/problemParameters.txt", "w")

            mat = scipy.io.loadmat(trainingData, squeeze_me=True)
            nFareClasses = mat['nFareClasses']
            fClassPrices = mat['fClassPrices']
            demandDistrType = mat['demandDistrType']
            interArrivalDistrType = mat['interArrivalDistrType']
            fClassCancelRate = mat['fClassCancelRate']
        
            g.write("nFareClasses:" + repr(nFareClasses) + "\n")
            g.write("fClassPrices:" + repr(fClassPrices) + "\n")
            g.write("demandDistributionType:" + repr(demandDistrType) + "\n")
            g.write("interArrival distribution type:" + repr(interArrivalDistrType) + "\n")            
            g.write("fclassCancellationRates:" + repr(fClassCancelRate) + "\n")
            
            g.write("targetModelUpdate:" + repr(target_model_update) + "\n")
            g.write("nTrEpisodes:" + repr(nTrEpisodes) + "\n")
            g.write("gamma:" + repr(gamma) + "\n")
            g.write("policyStr:" + repr(policyStr) + "\n")
            # f.write(repr(np.mean(aP)) + ", ")
            g.write("nbActions:" + repr(nb_actions) + "\n")
            g.write("nHiddenNeurons:" + repr(nHiddenNeurons) + "\n")
            g.write("nHiddenLayers:" + repr(nHiddenLayers) + "\n")
            g.write("wtpGroupMean:" + repr(fd) + "\n")
            g.write("cncSetting:" + repr(cnc) + "\n")
            g.write("bumpingCostFactor:" + repr(bumpingCostFactor) + "\n")
            g.write("avgOptimalReward_Test:" + repr(np.mean(rP)) + "\n")
            g.write("avgLoadFactor_Test:" + repr(np.mean(LF)) + "\n")
            g.write("trainingData filename:" + trainingData + "\n")
            g.write("testData filename:" + testData + "\n")

            g.write("Best model:" + repr(bestModelIndex+1) + ", ")
            g.write("mean RP of best model:" + repr(max(trModels_meanRP)) + ", ")            
            g.write("mean LF of best model:" + repr(trModels_meanLF[bestModelIndex]) + "\n")
            
            g.close()
##            
##
##            # average_rp = np.mean(rP)
##            # average_ap = np.mean(aP)
##            # average_pf = np.mean(LF)
##
##            # plot training
##            # pdb.set_trace()
##            plt.plot(t1, rP, t2, running_mean(rP, rmInterval), 'r')
##            # plt.title('Learning Curve')
##            plt.ylabel('Percentage of Optimal Reward')
##            plt.xlabel('Episode')
##            plt.savefig(outputDir + '/test_optimalRewardPercentage.png')
##            # plt.show()
##            plt.close()
##
##
####            plt.plot(t1, aP, t2, running_mean(aP, rmInterval), 'r')
####            # plt.title('Percentage of Arrivals Accepted')
####            plt.ylabel('Percentage of Arrivals Accepted')
####            plt.xlabel('Episode')
####            plt.savefig(outputDir + '/test_acceptancePercentage.png')
####            # plt.show()
####            plt.close()
##
##            plt.plot(t1, LF, t2, running_mean(LF, rmInterval), 'r')
##            # plt.title('Load factor')
##            plt.ylabel('Load factor')
##            plt.xlabel('Episode')
##            plt.savefig(outputDir + '/test_loadFactor.png')
##            # plt.show()
##            plt.close()
##
##            plt.plot(test_logger.seats)
##            plt.title('Seat Allocation')
##            plt.ylabel('Number of Seats per Class')
##            plt.xlabel('Episode')
##            plt.savefig(outputDir + '/test_seat_allocation.png')
##            #plt.show()
##            plt.close()
##
##            print('saving info_logger and test_logger objects to file by data serialization using pickle module')
##
####            with open(outputDir + '/trainAndTestLoggerObjs.pkl', 'wb') as pklFilePointer:  # Python 3: open(..., 'wb')
####                pickle.dump([info_logger, test_logger], pklFilePointer)
##            
####            with open(outputDir + '/trainLoggerObj.pkl', 'wb') as f:
####                pickle.dump(info_logger, f)
####
####            with open(outputDir + '/testLoggerObj.pkl', 'wb') as f:
####                pickle.dump(test_logger, f)
##
##            
##
##            with open(outputDir + '/trainLogger.pkl', 'wb') as pklFilePointer:
##                pickle.dump([info_logger.seats, info_logger.nSeatsSoldByPrices, info_logger.nCancellationsByPrices, info_logger.episode_reward,
##                             info_logger.max_reward, info_logger.rewardPercentage, info_logger.overbooking,
##                             info_logger.loadFactor, info_logger.observations, info_logger.rewards, info_logger.actions], pklFilePointer)
##
##            with open(outputDir + '/testLogger.pkl', 'wb') as pklFilePointer:
##                pickle.dump([test_logger.seats, test_logger.nSeatsSoldByPrices, test_logger.nCancellationsByPrices, test_logger.episode_reward,
##                             test_logger.max_reward, test_logger.rewardPercentage, test_logger.overbooking,
##                             test_logger.loadFactor, test_logger.observations, test_logger.rewards, test_logger.actions], pklFilePointer)

##                       
f.close()



