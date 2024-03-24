# This script is used to train and test the DRL agent with the air travel market simulator
# Author: Syed A.M. Shihab



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

from airTravelMarketSimulator_env import *

biased = True # bias is present
computeRewardAtEnd = False # (immediate) reward is calculated after each action

target_model_update = 1e-2 # 10000 # 1e-2 # 0.01  -> 0.01<1, so soft update of target model: target_model_params = target_model_update * local_model_params + (1 - target_model_update) * taget_model_params


nTrEpisodes = 25000 # it has to be some multiple of checkPointInterval 
indexSimulationRun = 1

gamma = 0.995
nb_eps_steps = 182
nb_steps = nb_eps_steps*nTrEpisodes  
logging_interval = nb_steps/10 
rmInterval = 111 


value_max=1
value_min=.01

policyStr = "LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.01, value_test=.05, nb_steps=nb_steps)"  
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.01, value_test=.05, nb_steps=nb_steps) # nb_steps=nb_steps-nb_eps_steps*500
# policy = EpsGreedyQPolicy(eps=.1)
# policy = LinearAnnealedPolicy(BoltzmannQPolicy(), attr='tau', value_max=1., value_min=.1, value_test=.05, nb_steps=nb_steps)
test_policy = GreedyQPolicy() # default testPolicy

arm_processor = armProcessor(biased)

nActions = 3 # 6 # 18 # 6 # 18

nHiddenNeuronsByLayer = [256, 256, 256, 64] # [256, 256] # [1024, 1024, 1024] # [16, 16, 16] # [512, 512, 512, 64] # [256, 256, 128, 64] # [512, 512, 512, 64] # [256, 256, 128, 64] 

nHiddenNeurons = 256
nHiddenLayers = 3


# Building neural network model TODO: neural network hyperparameter optimization
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
performanceLogDir = "numericalExperimentsResults" # location where results.txt file (containing performance metrics values of each simulation run) is saved
if not os.path.isdir(performanceLogDir): 
    os.mkdir(performanceLogDir)

f = open(performanceLogDir+"/results.txt", "w")
f.write("bumpingCostFactor, cancelProbsSetting, meanNarrivalsSetting, test_mean RP of best model, test_mean LF of best model" + "\n")


# Running numerical experiments for sensitivity analysis
bcfArray = [2] # [2.0, 2.5]
cncArray = [1,2,3] # [1] # [ 2 3]
fdArray = [1,2,3] # [1 2 3] # fdArray = [3] # changed from fdArray = [1] 
for bumpingCostFactor in bcfArray:
    for cancelProbsSetting in cncArray:
        for meanNarrivalsSetting in fdArray:

            
            outputDir = "numExp_outputs_" + "OB" + repr(bumpingCostFactor) + "cnc" + repr(cancelProbsSetting) + "_fcArrivals" + repr(meanNarrivalsSetting) + "_" + repr(indexSimulationRun) # location where output plots and model weights are saved # TODO: change last digit
            if not os.path.isdir(outputDir):
                os.mkdir(outputDir)


            # Creating DQN agent
            dqn = DQNAgent(processor=arm_processor, model=model, test_policy=test_policy, nb_actions=nActions,
                           memory=memory, policy=policy, target_model_update=target_model_update, gamma=gamma) # gamma = discount factor

            dqn.compile(Adam(lr=1e-3), metrics=['mae']) # lr=.00025

            ######## Specify output directory and model number #################
            # loading weights of the model obtained by training the agent on deterministic market settings; these weights act as a good initial starting weight vector for training
            # or, loading weights of the best model for trainingData = training_c1_fd1_startTr5 when cancellationRate = [0, 0]
            # dqn.load_weights('output_OB2cnc1fd1_53' + '/dqn_armEnv_weights_model358.h5f')
                   
            trainingData = "trainingData_cnc" + repr(cancelProbsSetting) + "_fcArrivals" + repr(meanNarrivalsSetting) + ".mat"  # 'training_c1_fd1_startTr' # 'training_c1_fd1_testDP6' # 'training_c1_fd1_testDP5.mat'

            env = armEnv(trainingData, biased, computeRewardAtEnd, bumpingCostFactor) # creating armEnv instance

            info_logger = infoLogger()
            test_logger = infoLogger()


            # save weights and test model after every 500 episodes
            nTestEpisodes = 300
            stepsCounter = 0
            tr_RP = [] # records RP of agent in all training episodes
            tr_LF = [] # records LF of agent in all training episodes
            trModels_testPerf_meanRP = [] # records mean RP (test performance) of agent in test episodes at all the checkpoints 
            trModels_testPerf_meanLF = [] # records mean LF (test performance) of agent in test episodes at all the checkpoints 
            checkPointInterval = 250
            while (stepsCounter<nb_steps):
                                
                # Run training for 250 episodes (= 250*182 steps)
                history = dqn.fit(env, callbacks=[info_logger], verbose=0, nb_steps=checkPointInterval*182, log_interval=logging_interval) 

                stepsCounter += checkPointInterval*182
                value_max = max(value_min, 1-((1-value_min)/nb_steps)*stepsCounter) # value_max = policy.get_current_value()
                dqn.policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=value_max, value_min=.01, value_test=.05, nb_steps=(nb_steps-stepsCounter)) # nb_steps=nb_steps-nb_eps_steps*500
                
                tr_RP.append(info_logger.rewardPercentage)
                tr_LF.append(info_logger.loadFactor)
                


                # Run test
                dqn.test(env, callbacks = [test_logger], nb_episodes=nTestEpisodes, visualize=False)
                iterInd = int((stepsCounter/(checkPointInterval*182)))                
                # rP = np.array(test_logger.rewardPercentage[iterInd*300-300:iterInd*300])*100
                rP = np.array(test_logger.rewardPercentage)*100
                trModels_testPerf_meanRP.append(np.mean(rP))                
                # LF = np.array(test_logger.loadFactor[iterInd*300-300:iterInd*300])*100
                LF = np.array(test_logger.loadFactor)*100
                trModels_testPerf_meanLF.append(np.mean(LF))
                print('mean RP of current model =', np.mean(rP))
                print('mean LF of current model =', np.mean(LF))

                print('current episode index =', iterInd*checkPointInterval)
                print('epsilon value_max =', value_max)

                print('cancelProbsSetting =', cancelProbsSetting)
                print('meanNarrivalsSetting =', meanNarrivalsSetting)

                dqn.save_weights(outputDir + '/dqn_armEnv_weights_model{}.h5f'.format(iterInd), overwrite=True)
                # dqn.load_weights(outputDir + '/dqn_armEnv_weights_model{}.h5f'.format(iterInd))
                # weights = dqn.model.get_weights()





            bestModelIndex = trModels_testPerf_meanRP.index(max(trModels_testPerf_meanRP))
            print('Best model = ' + repr(bestModelIndex+1))
            print('mean RP of best model = ' + repr(max(trModels_testPerf_meanRP)))
            print('mean LF of best model = ' + repr(trModels_testPerf_meanLF[bestModelIndex]))
            



            nModels = len(trModels_testPerf_meanRP) # 300
            rmInterval = 55 # 33 # 11

            t1 = np.arange(nModels)+1
            t2 = np.arange(1+(rmInterval-1)/2, nModels+1-(rmInterval-1)/2)
            
            # Making plots of rP of trModels at the checkpoints
            plt.plot(t1, trModels_testPerf_meanRP, t2, running_mean(trModels_testPerf_meanRP, rmInterval))
            # plt.title('Learning Curve')
            plt.ylabel('Optimal Revenue (%)')
            plt.xlabel('Model')
            plt.savefig(outputDir + '/trPlot_modelsAtCheckpoints_optimalRP.png')
            plt.close()
            # plt.show()

            # Making plots of lf of trModels at the checkpoints
            plt.plot(t1, trModels_testPerf_meanLF, t2, running_mean(trModels_testPerf_meanLF, rmInterval))
            plt.ylabel('Load Factor (%)')
            plt.xlabel('Model')
            plt.savefig(outputDir + '/trPlot_modelsAtCheckpoints_LF.png')
            plt.close()
            # plt.show()            
                



            
            # loading best model, testing its performance, and saving its testLogger object
            dqn.load_weights(outputDir + '/dqn_armEnv_weights_model{}.h5f'.format(bestModelIndex+1)) 
                        
            testData = "testData_cnc" + repr(cancelProbsSetting) + "_fcArrivals" + repr(meanNarrivalsSetting) + ".mat"  
            env = armEnv(testData, biased, computeRewardAtEnd, bumpingCostFactor) # creating armEnv instance

            test_logger = infoLogger()
            nTestEpisodes = 300 # 150
            rmInterval = int(nTestEpisodes/10)
                       
            dqn.test(env, callbacks = [test_logger], nb_episodes=nTestEpisodes, visualize=False)

            # nTestEpisodes = len(rP)
            t1 = np.arange(nTestEpisodes)+1
            t2 = np.arange(1+(rmInterval-1)/2, nTestEpisodes+1-(rmInterval-1)/2)
            rP = np.array(test_logger.rewardPercentage)*100
            # aP = np.array(test_logger.acceptPercentage)*100
            LF = np.array(test_logger.loadFactor)*100
            # np.savez(outputDir + '/TrainingData.npz', rP=rP, aP=aP, LF=LF)

            print('avgOptimalRewardPercentage of best model = ' + repr(np.mean(rP)))
            # print('average_ap = ' + repr(np.mean(aP)))
            print('avgLoadFactor of best model = ' + repr(np.mean(LF)))

            # saving the test_logger object of the best model found in this numerical experiment
            with open(outputDir + '/testLogger_bestModel.pkl', 'wb') as pklFilePointer1:
                pickle.dump([test_logger.nBookings, test_logger.nCancellations, test_logger.episode_reward,
                             test_logger.max_reward, test_logger.rewardPercentage, test_logger.overbooking,
                             test_logger.loadFactor, test_logger.observations, test_logger.rewards, test_logger.actions], pklFilePointer1)   

            # saving tr_RP, tr_LF, trModels_testPerf_meanRP and trModels_testPerf_meanLF
            with open(outputDir + '/trRP_trLF_modelsTestRP_modelsTestLF.pkl', 'wb') as pklFilePointer2:
                pickle.dump([tr_RP, tr_LF, trModels_testPerf_meanRP, trModels_testPerf_meanLF], pklFilePointer2)            




            # writing simulation/experiment details onto files problemParameters.txt and results.txt
            f.write(repr(bumpingCostFactor) + ", ")
            f.write(repr(cancelProbsSetting) + ", ")
            f.write(repr(meanNarrivalsSetting) + ", ")
            f.write(repr(trModels_testPerf_meanRP[bestModelIndex]) + ", ")
            f.write(repr(trModels_testPerf_meanLF[bestModelIndex]) + "\n")
            
            g = open(outputDir+"/problemParameters.txt", "w")            
            
            mat = scipy.io.loadmat(trainingData, squeeze_me=True)
            nFareClasses = mat['nFareClasses']
            fClassPrices = mat['fClassPrices']
            fClassMeanArrivals = mat['fClassMeanArrivals']
            arrivalProcess = 'NHPP'
            cncProb_fClasses = mat['cncProb_fClasses']
            cancellationProcess = 'time-varying cancellation probabilities'
            
            g.write("fcArrivalsSetting:" + repr(meanNarrivalsSetting) + "\n")
            g.write("cancellationProbSetting:" + repr(cancelProbsSetting) + "\n")
            g.write("bumpingCostFactor:" + repr(bumpingCostFactor) + "\n")
            
            g.write("nFareClasses:" + repr(nFareClasses) + "\n")
            g.write("fClassPrices:" + repr(fClassPrices) + "\n")
            g.write("fClassMeanArrivals:" + repr(fClassMeanArrivals) + "\n")  
            g.write("arrivalProcess:" + repr(arrivalProcess) + "\n")
            g.write("cancellationProcess:" + repr(cancellationProcess) + "\n")            
            g.write("cncProb_fClasses_last2timeSteps:" + repr(cncProb_fClasses[:,-2:]) + "\n")
            g.write("cncProb_fClasses_remainingtimeSteps:" + repr(cncProb_fClasses[:,1]) + "\n")
            
            g.write("targetModelUpdate:" + repr(target_model_update) + "\n")
            g.write("nTrEpisodes:" + repr(nTrEpisodes) + "\n")
            g.write("gamma:" + repr(gamma) + "\n")
            g.write("policyStr:" + repr(policyStr) + "\n")
            # f.write(repr(np.mean(aP)) + ", ")
            g.write("nbActions:" + repr(nActions) + "\n")
            g.write("nHiddenNeurons:" + repr(nHiddenNeurons) + "\n")
            g.write("nHiddenLayers:" + repr(nHiddenLayers) + "\n")
            
            
            g.write("trainingData filename:" + trainingData + "\n")
            g.write("testData filename:" + testData + "\n")

            g.write("Best model:" + repr(bestModelIndex+1) + ", ")
            g.write("mean RP of best model:" + repr(trModels_testPerf_meanRP[bestModelIndex]) + ", ")            
            g.write("mean LF of best model:" + repr(trModels_testPerf_meanLF[bestModelIndex]) + "\n")
            
            g.close()

f.close()



