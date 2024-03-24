# TODO: change armClasses...py name if necessary, trainingData filename (before training and before testing), nbActions, nerual network input, nHiddenNeurons, nTestEpisodes if necessary after
# changing market parameters
# TODO: change nTrEpisodes and indexSimulationRun before each new run
# TODO: change the filename (modelIndex and simulationRunIndex) of the best model before loading weights onto the neural network

# list of hyperparameters: target_model_update, nTrEpisodes, gamma, exploration policy, \
# nHiddenNeuronsEachLayer, nHiddenLayers, activation function, lr, DRL algorithm (DQL, double DQL, dueling DQL, etc.) \
# optimizer (Adam, RMSprop, etc.) dqn.compile metrics (mae), 

import numpy as np
import scipy.io
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

#############################################################################################################################################################################
nTrEpisodes = 10000 # 250 # 500 # 10000 # 30000 # 1000 # 10000 # 100000 # 20000 # 99000 # 10000 # 10000 # 49000 # 10000 # 20000
#############################################################################################################################################################################

# MDP characteristics
gamma = 0.9999 # TODO: CHANGED 0.995^182 ~= 0.4
nStepsPerEps = 182
nb_steps = nStepsPerEps*nTrEpisodes  # 10,000 episodes; TODO: vary, try 182x20000 -> 20000 episodes; number of steps/state transitions in each episode = bookingHorizon = 182

logging_interval = nb_steps/10 # TODO: check convergence criteria/method/mechanism/strategy
rmInterval = 55 # 11 # 111 # 333 # 555 # choose odd running mean interval # int(nb_steps/1000) # TODO: use replay buffer


# Setting exploration policy
value_max=1
value_min=0 # .01
# policyStr = "LinearAnnealedPolicy(BoltzmannQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=nb_steps)"
policyStr = "LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=0, value_test=.05, nb_steps=nb_steps)"                              
policy1 = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=value_max, value_min=value_min, value_test=.05, nb_steps=nb_steps) # nb_steps=nb_steps-nStepsPerEps*500
policy2 = LinearAnnealedPolicy(BoltzmannQPolicy(), attr='tau', value_max=value_max, value_min=value_min, value_test=.05, nb_steps=nb_steps)
# policy3 = EpsGreedyQPolicy(eps=.1)

# Setting test policy
test_policy = GreedyQPolicy() # default testPolicy

arm_processor = armProcessor(biased)
nb_actions = 6 # 18 # 6 # 18
nStateVars = 1+2+2+2 # 7

# specifying directory where results of numerical experiments will be logged
dir_numExpsResults = "numericalExperimentsResults_2fp22pp_twg1wg4c1c2" # location where results.txt file (containing performance metrics values of each simulation run) is saved
if not os.path.isdir(dir_numExpsResults): 
    os.mkdir(dir_numExpsResults)

f = open(dir_numExpsResults+"/results.txt", "w")
f.write("nHiddenLayers, nHiddenNeurons, policy, target model update, learning rate, mean RP (%), std RP (%), mean LF (%), std LF (%)" + "\n")


# Running numerical experiments
expIndex = 1
nHiddenNeurons_optionsArray = [64, 128, 256] # [32,64,128] # [32,64,128,256] 
nHiddenLayers_optionsArray = [2,3] # [1,2,3,4] # [16,32,64,128] 
target_model_update_optionsArray = [1e-2, 10000] # [1e-2] 
# target_model_update = 1e-2 # 10000 # 1e-2 # 0.01  -> 0.01<1, so soft update of target model: target_model_params = target_model_update * local_model_params + (1 - target_model_update) * taget_model_params
# TODO: try target_model_update = 10,000 (hard update every 10000th step)
policy_optionsArray = [1] # [1, 2]
lr_optionsArray = [1e-3, 1e-4] # [1e-2,1e-3,1e-4]


for nHiddenLayers in nHiddenLayers_optionsArray:
    for nHiddenNeurons in nHiddenNeurons_optionsArray:
        for target_model_update in target_model_update_optionsArray:
            for policyInd in policy_optionsArray:
                for lr in lr_optionsArray:

                    if policyInd==1:
                        policy=policy1
                    else:
                        policy=policy2

                    tmuIndex = target_model_update_optionsArray.index(target_model_update)
                    lrIndex = lr_optionsArray.index(lr)

                    #############################################################################################################################################################################
                    expLabel = 'nHL' + repr(nHiddenLayers) + 'nHN' + repr(nHiddenNeurons) + 'tmu' + repr(tmuIndex) + 'lr' + repr(lrIndex) + 'pol1' # nHiddenLayers = 3; nHiddenNeurons = 16 # TODO:
                    #############################################################################################################################################################################

                    


                    #############################################################################################################################################################################

                    ######## Specify Output directory and model number #################
                    # loading weights of the best model for trainingData = training_c1_fd1_startTr5 when cancellationRate = [0, 0]
                    # outputDir = "output_OBcnc1fd1_2fp22pp_twg1wg4c1c2_1000" # location where output plots and model weights are saved # TODO: change last digit
                    ##bestModelIndex = 3
                    ##dqn.load_weights('output_OB2cnc1fd1_2fp22pp_twg1wg4c1c2_1001' + '/dqn_armEnv_weights_model{}.h5f'.format(bestModelIndex+1))

                    #############################################################################################################################################################################


                    # Varying bumping cost factor, cncProb_wtpGrp_t and arrivalRate_wtpGrp_t
                    bcfArray = [2] # [2.0, 2.5]
                    cncArray = [1] # [ 2 3]
                    fdArray = [1] # [1 2 3] # fdArray = [3] # changed from fdArray = [1] 
                    for bumpingCostFactor in bcfArray:
                        for cnc in cncArray:
                            for fd in fdArray:

                                ##activationFunc = 'relu'

                                # Build model TODO: neural network hyperparameter optimization
                                model = Sequential()
                                
                                if(biased):
                                    model.add(Flatten(input_shape=(1,nStateVars+1))) # 8 # 12 # input to NN: ([time, seats[0], seats[1], seats[2], bias=1]) array
                                else:
                                    model.add(Flatten(input_shape=(1,nStateVars))) # 7 # 11

                                for ind in range(nHiddenLayers):
                                    model.add(Dense(nHiddenNeurons)) # 128 # 16, 48 gives average_rp ~ 80% only # (16,16,16), (256,256), (256,256,128)
                                    model.add(Activation('relu'))

                                model.add(Dense(nb_actions))
                                model.add(Activation('linear'))


                                memory = SequentialMemory(limit=50000, window_length=1)

                                dqn = DQNAgent(processor=arm_processor, model=model, test_policy=test_policy, nb_actions=nb_actions,
                                        memory=memory, policy=policy, target_model_update=target_model_update, gamma=gamma) # gamma = discount factor

                                dqn.compile(Adam(lr=lr), metrics=['mae']) # lr=.00025 in RMSprop (used by Mnih)


                                

                                
                                outputDir = "output_" + "OB" + repr(bumpingCostFactor) + "cnc" + repr(cnc) + "fd" + repr(fd) + "_2fp22pp_twg1wg4c1c2_" + expLabel + repr(expIndex) # location where output plots and model weights are saved # TODO: change last digit
                                if not os.path.isdir(outputDir):
                                    os.mkdir(outputDir)

                                #############################################################################################################################################################################
                                    
                                # trainingData = "training_c" + repr(cnc) + "_fd" + repr(fd) + "_startTr7" + ".mat"  # 'training_c1_fd1_startTr' # 'training_c1_fd1_testDP6' # 'training_c1_fd1_testDP5.mat'
                                trainingData = "trData_cnc" + repr(cnc) + "_fcArrivals" + repr(fd) + "_2fp22pp_mktSiml_mktEst" + ".mat"  # 'training_c1_fd1_startTr' # 'training_c1_fd1_testDP6' # 'training_c1_fd1_testDP5.mat'
                                # fileName = 'trData_cnc' + string(cancelProbsSetting) + '_fcArrivals' + string(meanNarrivalsSetting) + '_2fp22pp_mktSiml_mktEst'; 

                                #############################################################################################################################################################################
                                
                                env = armEnv(trainingData, biased, computeRewardAtEnd, bumpingCostFactor) # creating armEnv instance

                                info_logger = infoLogger()
                                test_logger = infoLogger()
                                

                                # save weights and test model after every 250 episodes
                                nTestEpisodes = 300
                                stepsCounter = 0
                                tr_RP = [] # records RP of agent in all training episodes
                                tr_LF = [] # records LF of agent in all training episodes
                                trModels_testPerf_meanRP = [] # records mean RP (test performance) of agent in test episodes at all the checkpoints
                                trModels_testPerf_stdRP = [] # records std RP (test performance) of agent in test episodes at all the checkpoints
                                trModels_testPerf_meanLF = [] # records mean LF (test performance) of agent in test episodes at all the checkpoints
                                trModels_testPerf_stdLF = [] # records std LF (test performance) of agent in test episodes at all the checkpoints
                                
                                #############################################################################################################################################################################
                                checkPointInterval = 250 # 125 # 250
                                #############################################################################################################################################################################
                                convergence_eps_005 = []
                                convergence_eps_01 = []
                                convergence_eps_02 = []
                                convergence_eps_05 = []
                                
                                while (stepsCounter<nb_steps):                                    

                                    if stepsCounter>=(nStepsPerEps*(nTrEpisodes-500)): # last 500 episodes
                                        # Continuing training with epsilon=0 or greedyQpolicy for the last 500 episodes
                                        dqn.policy = GreedyQPolicy()
                                        history = dqn.fit(env, callbacks=[info_logger], verbose=0, nb_steps=checkPointInterval*182, log_interval=logging_interval) 
                                        stepsCounter += checkPointInterval*182
                                    else:
                                        # Run training for 250 episodes (= 250*182 steps)
                                        history = dqn.fit(env, callbacks=[info_logger], verbose=0, nb_steps=checkPointInterval*182, log_interval=logging_interval)
                                        stepsCounter += checkPointInterval*182
                                        value_max = max(value_min, 1-((1-value_min)/nb_steps)*stepsCounter) # value_max = policy.get_current_value()
                                        dqn.policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=value_max, value_min=.01, value_test=.05, nb_steps=(nb_steps-stepsCounter)) # nb_steps=nb_steps-nStepsPerEps*500
                                    
                                    tr_RP.append(info_logger.rewardPercentage)
                                    tr_LF.append(info_logger.loadFactor)
                                    
                    ##                # Save neural network weights
                    ##                dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
                    ##
                    ##                # Save memory
                    ##                pickle.dump(memory, open("memory.pkl", "wb"))

                                    # Testing the latest policy/NN
                                    dqn.test(env, callbacks = [test_logger], nb_episodes=nTestEpisodes, visualize=False)
                                    iterInd = int((stepsCounter/(checkPointInterval*182)))                
                                    
                                    rP = np.array(test_logger.rewardPercentage)*100
                                    trModels_testPerf_meanRP.append(np.mean(rP))
                                    trModels_testPerf_stdRP.append(np.std(rP))
                                    
                                    LF = np.array(test_logger.loadFactor)*100
                                    trModels_testPerf_meanLF.append(np.mean(LF))
                                    trModels_testPerf_stdLF.append(np.std(LF))

                                    print('mean RP of current model =', np.mean(rP))
                                    print('mean LF of current model =', np.mean(LF))

                                    print('current episode index =', iterInd*checkPointInterval)
                                    print('epsilon value_max =', value_max)

                                    print('expLabel: ',expLabel, ' expIndex: ',expIndex)
                                    print(model.summary())

                                    # Saving the current neural network weights (current policy)
                                    dqn.save_weights(outputDir + '/dqn_armEnv_weights_model{}.h5f'.format(iterInd), overwrite=True)                                    

                                    # checking for convergence at this checkpoint
                                    rm_tr_RP = running_mean(tr_RP, 11) # rmInterval = 11
                                    diff_rm_tr_RP = [abs(rm_tr_RP[-1]-rm_tr_RP[-ind]) for ind in range(2,12)]
                                    if max(diff_rm_tr_RP)<=0.005:
                                        convergence_eps_005.append(iterInd*checkPointInterval) 
                                    if max(diff_rm_tr_RP)<=0.01:
                                        convergence_eps_01.append(iterInd*checkPointInterval) 
                                    if max(diff_rm_tr_RP)<=0.02:
                                        convergence_eps_02.append(iterInd*checkPointInterval) 
                                    if max(diff_rm_tr_RP)<=0.05:
                                        convergence_eps_05.append(iterInd*checkPointInterval)
                                    # alternative: epsilon optimal stopping criterion - max difference in Q(s,a) values < epsilon (tolerance)
                                    # epsilon (float) – Stopping criterion. The maximum change in the value function at each iteration is compared against epsilon.
                                    # Once the change falls below this value, then the value function is considered to have converged to the optimal value function. 
                                    
                                

                                bestModelIndex = trModels_testPerf_meanRP.index(max(trModels_testPerf_meanRP))
                                print('Best model = ' + repr(bestModelIndex+1))
                                print('mean RP of best model = ' + repr(max(trModels_testPerf_meanRP)))
                                print('mean LF of best model = ' + repr(trModels_testPerf_meanLF[bestModelIndex]))


                                # Generating plots
                                nModels = len(trModels_testPerf_meanRP) # 120
                                # rmInterval = 55 # 33 # 11

                                t1 = np.arange(nModels)+1
                                t2 = np.arange(1+(rmInterval-1)/2, nModels+1-(rmInterval-1)/2)

                                t1_eps = np.arange(nTrEpisodes)+1
                                t2_eps = np.arange(1+(rmInterval-1)/2, nTrEpisodes+1-(rmInterval-1)/2)
                                
                                # Generating RP vs trModel (at the checkpoints) plot
                                plt.plot(t1, trModels_testPerf_meanRP, t2, running_mean(trModels_testPerf_meanRP, rmInterval))
                                # plt.title('Learning Curve')
                                plt.ylabel('Optimal Revenue (%)')
                                plt.xlabel('Model')
                                plt.savefig(outputDir + '/trPlot_modelsAtCheckpoints_optimalRP.pdf')
                                plt.close()
                                # plt.show()

                                # tr_RP = np.array(tr_RP)*100
                                flat_tr_RP = [item for sublist in np.array(tr_RP)*100 for item in sublist]
                                # tr_LF = np.array(tr_LF)*100
                                flat_tr_LF = [item for sublist in np.array(tr_LF)*100 for item in sublist]
                                
                                # Generating RP vs episode plot 
                                plt.plot(t1_eps, flat_tr_RP, t2_eps, running_mean(flat_tr_RP, rmInterval))
                                # plt.title('Learning Curve')
                                plt.ylabel('Optimal Revenue (%)')
                                plt.xlabel('Episode')
                                plt.savefig(outputDir + '/trPlot_episode_optimalRP.pdf')
                                plt.close()
                                # plt.show()

                                # Generating LF vs trModel (at the checkpoints) plot
                                plt.plot(t1, trModels_testPerf_meanLF, t2, running_mean(trModels_testPerf_meanLF, rmInterval))
                    ##            plt.plot(trModels_testPerf_meanLF)
                                # plt.title('Learning Curve')
                                plt.ylabel('Load Factor (%)')
                                plt.xlabel('Model')
                                plt.savefig(outputDir + '/trPlot_modelsAtCheckpoints_LF.pdf')
                                plt.close()
                                # plt.show()

                                # Generating LF vs episode plot 
                                plt.plot(t1_eps, flat_tr_LF, t2_eps, running_mean(flat_tr_LF, rmInterval))
                                # plt.title('Learning Curve')
                                plt.ylabel('Load Factor (%)')
                                plt.xlabel('Episode')
                                plt.savefig(outputDir + '/trPlot_episode_LF.pdf')
                                plt.close()
                                # plt.show()
                                    

                                ### saving the testLogger object created by the best model
                                
                                # loading the weights of the best model
                                dqn.load_weights(outputDir + '/dqn_armEnv_weights_model{}.h5f'.format(bestModelIndex+1)) 

                                # testing best model in mktSiml_mktEst using trData (the env defined above)
                                test_logger = infoLogger()
                                nTestEpisodes = 300
                                # rmInterval = int(nTestEpisodes/10)

                                dqn.test(env, callbacks = [test_logger], nb_episodes=nTestEpisodes, visualize=False)

                                # nTestEpisodes = len(rP)
                    ##            t1 = np.arange(nTestEpisodes)+1
                    ##            t2 = np.arange(1+(rmInterval-1)/2, nTestEpisodes+1-(rmInterval-1)/2)
                                RP = np.array(test_logger.rewardPercentage)*100
                                LF = np.array(test_logger.loadFactor)*100
                                # np.savez(outputDir + '/TrainingData.npz', RP=RP, aP=aP, LF=LF)

                                print('avg optimal reward percentage of best model = ' + repr(np.mean(RP)))
                                # print('average_ap = ' + repr(np.mean(aP)))
                                print('avg load factor of best model = ' + repr(np.mean(LF)))



                    #############################################################################################################################################################################

                    ##            # testing the best model in the true market simulator
                    ##            # saving the testLogger object created by the best model
                    ##            # specify best model
                    ##            policy = GreedyQPolicy() # agent will continue to learn (using dqn.fit()), but it is not allowed to explore (TODO: will letting it explore lead to good results in the long run)
                                # TODO: tune the learning rate lr # # CHANGED gamma = 1
                    ##            dqn = DQNAgent(processor=arm_processor, model=model, test_policy=test_policy, nb_actions=nb_actions,
                    ##                           memory=memory, policy=policy, target_model_update=target_model_update, gamma=1) 
                    ##
                    ##            dqn.compile(Adam(lr=1e-3), metrics=['mae']) # lr=.00025
                    ##
                    ##            dqn.load_weights(outputDir + '/dqn_armEnv_weights_model{}.h5f'.format(bestModelIndex+1)) 
                    ##            
                    ##            test_logger = infoLogger()
                    ##            nTestEpisodes = 2000 # 5000
                    ##            rmInterval = int(nTestEpisodes/10)
                    ##
                    ##            testData = "testData_cnc1_fcArrivals1_2fp22pp_true_mktSiml" + ".mat"            
                    ##            env = armEnv(testData, biased, computeRewardAtEnd, bumpingCostFactor) # creating armEnv instance
                    ##            
                    ####            dqn.test(env, callbacks = [test_logger], nb_episodes=nTestEpisodes, visualize=False)
                    ##            history = dqn.fit(env, callbacks=[test_logger], verbose=0, nb_steps=nTestEpisodes*182, log_interval=500) 
                    ##            
                    ##            # nTestEpisodes = len(RP)
                    ##            t1 = np.arange(nTestEpisodes)+1
                    ##            t2 = np.arange(1+(rmInterval-1)/2, nTestEpisodes+1-(rmInterval-1)/2)
                    ##            RP = np.array(test_logger.rewardPercentage)*100
                    ##            LF = np.array(test_logger.loadFactor)*100
                    ##
                    ##            print('avgOptimalRewardPercentage of best model in the last 500 episodes = ' + repr(np.mean(RP[-500:])))
                    ##            print('avgLoadFactor of best model in the last 500 episodes = ' + repr(np.mean(LF[-500:])))
                    ##
                    ##            # saving the test_logger object of the best model 
                    ##            with open(outputDir + '/testLogger_bestModel.pkl', 'wb') as pklFilePointer1:
                    ##                pickle.dump([test_logger.nBookings_fp, test_logger.nBookings_price, test_logger.preBumping_nBookings_fp, test_logger.preBumping_nBookings_price, test_logger.bumpingCost,
                    ##                             test_logger.nCancellations_price, test_logger.episode_reward, test_logger.max_reward, test_logger.rewardPercentage, test_logger.overbooking,
                    ##                             test_logger.loadFactor, test_logger.observations, test_logger.rewards, test_logger.actions], pklFilePointer1)

                    #############################################################################################################################################################################

                                    

                                # saving tr_RP, tr_LF, trModels_testPerf_meanRP and trModels_testPerf_meanLF
                                with open(outputDir + '/trRP_trLF_modelsTestRP_modelsTestLF.pkl', 'wb') as pklFilePointer2:
                                    pickle.dump([tr_RP, tr_LF, trModels_testPerf_meanRP, trModels_testPerf_stdRP, trModels_testPerf_meanLF, trModels_testPerf_stdLF, convergence_eps_005, convergence_eps_01, convergence_eps_02, convergence_eps_05], pklFilePointer2)

                                                                
                                
                    ##            # TODO: dqn.load_weights() (and also load memory using pickle if you are creating a new agent using dqn=dqnAgent(...)
                    ##            # two stage annealing process
                    ##            dqn.policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.01, value_min=.001, value_test=.05, nb_steps=nb_steps) # nb_steps=nb_steps-nStepsPerEps*500
                    ##            history = dqn.fit(env, callbacks=[info_logger], verbose=0, nb_steps=500, log_interval=logging_interval) 


                                
                                # writing simulation/experiment details onto files problemParameters.txt and results.txt
                                f.write(repr(nHiddenLayers) + ", ")
                                f.write(repr(nHiddenNeurons) + ", ")
                                f.write(repr(policyInd) + ", ")
                                f.write(repr(target_model_update) + ", ")
                                f.write(repr(lr) + ", ")
                                f.write("{:.2f}".format(np.mean(RP)) + ", ")
                                f.write("{:.2f}".format(np.std(RP)) + ", ")
                                f.write("{:.2f}".format(np.mean(LF)) + ", ")
                                f.write("{:.2f}".format(np.std(LF)) + "\n ")
                                           
                                
                                g = open(outputDir+"/problemParameters.txt", "w")        
                                
                                mat = scipy.io.loadmat(trainingData, squeeze_me=True)
                                nFareProducts = mat['nFareProducts']
                                fareLevels = mat['fareLevels']
                                # wtpGroupMeanArrivals = mat['wtpGroupMeanArrivals']
                                arrivalProcess = 'NHPP'
                                # cncProb_fClasses = mat['wtpGroupCncRate_last2timeSteps_allExp']
                                cancellationProcess = 'time-varying cancellation probabilities'
                                
                                g.write("meanArrivalSetting:" + repr(fd) + "\n")
                                g.write("cancellationProbSetting:" + repr(cnc) + "\n")
                                g.write("bumpingCostFactor:" + repr(bumpingCostFactor) + "\n")
                                
                                g.write("nFareProducts:" + repr(nFareProducts) + "\n")
                                g.write("fareLevels:" + repr(fareLevels) + "\n")
                                # g.write("wtpGroupMeanArrivals:" + repr(wtpGroupMeanArrivals) + "\n")  
                                g.write("arrivalProcess:" + repr(arrivalProcess) + "\n")
                                g.write("cancellationProcess:" + repr(cancellationProcess) + "\n")            
                                # g.write("cncProb_fClasses_last2timeSteps:" + repr(cncProb_fClasses[:,-2:]) + "\n")
                                # g.write("cncProb_fClasses_remainingtimeSteps:" + repr(cncProb_fClasses[:,1]) + "\n")
                                
                                g.write("targetModelUpdate:" + repr(target_model_update) + "\n")
                                g.write("nTrEpisodes:" + repr(nTrEpisodes) + "\n")
                                g.write("gamma:" + repr(gamma) + "\n")
                                g.write("policyStr:" + repr(policyStr) + "\n")
                                # f.write(repr(np.mean(aP)) + ", ")
                                g.write("nbActions:" + repr(nb_actions) + "\n")
                                g.write("nHiddenNeurons:" + repr(nHiddenNeurons) + "\n")
                                g.write("nHiddenLayers:" + repr(nHiddenLayers) + "\n")
                                
                    ##            g.write("mean RP of best model:" + repr(np.mean(rP)) + "\n")
                    ##            g.write("mean LF of best model:" + repr(np.mean(LF)) + "\n")
                                
                                g.write("trainingData filename:" + trainingData + "\n")
                                # g.write("testData filename:" + testData + "\n")

                                g.write("Best model:" + repr(bestModelIndex+1) + ", ")
                                g.write("mean RP of best model:" + repr(trModels_testPerf_meanRP[bestModelIndex]) + ", ")            
                                g.write("mean LF of best model:" + repr(trModels_testPerf_meanLF[bestModelIndex]) + "\n")
                                
                                g.close()

                                expIndex += 1
                                
f.close()



