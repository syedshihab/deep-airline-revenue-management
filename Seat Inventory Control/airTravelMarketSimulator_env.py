# This script is used to build the air travel market simulator (the ARM enviornment)
# Author: Syed A.M. Shihab

import numpy as np
from gym import spaces
import scipy.io

from keras.callbacks import Callback
from rl.core import Processor, Env



# implementing the armEnv environment
class armEnv(Env): 
    

    # filename - name of the MATLAB file containing the passenger arrival data and market characteristics    
    def __init__(self, filename, biased=1, computeRewardAtEnd=0, ob=2): 
        self.biased = biased
        self.computeRewardAtEnd = computeRewardAtEnd
        self.overbooking_cost_multiplier = ob # bumping cost factor
        self.reward_range = (-np.inf, np.inf) # immediate reward
        
        mat = scipy.io.loadmat(filename, squeeze_me=True) # load training data (passenger arrival data) generated with MATLAB
          
        # open/close fare class 
        self.action_space = spaces.Discrete(3) 

        self.actionMatrix = mat['actionMatrix']
        self.nFareClasses = mat['nFareClasses']
        self.capacity = mat['capacity']
        
        self.totalTime = mat['bookingHorizon'] 
        self.nEpisodes = mat['nEpisodes']-1
        
        self.cncFee = [0,0] 
        self.fClassPrices = mat['fClassPrices']
        self.trDataPy = mat['trDataPy'] 
        self.nArrivals_eachEps = mat['nArrivals_eachEps']
        self.currentEpisodeIndex = None
        self.currentEpisode = None
        self.nArrivalsCurrentEps = None 

        self.max_reward_list = mat['maxReward'] 
        self.max_reward = 0

        self.paxIndex = 0 # passenger index
        self.nTotalPax = 0 # number of passengers in a certain episode = self.currentEpisodeNArrivals
        self.time = 0 # episode current time; or time remaining till flight departure (end of episode)
        # when reading in data subtract 1 so that zero indexed
        
        self.nBookings = np.zeros(self.nFareClasses, dtype=int) # state variables: b1,b2,b3                
        self.nCancellations = np.zeros((self.nFareClasses-1), dtype=int) # nCancellations = c1, c2; [0 0]; cancellation rate of last fare class = 0
                
        self.cancellations = []

        self.done = False

                
        self.nStateVars = 1 + self.nFareClasses 
        self.observation_space = spaces.Box(low=np.zeros(self.nStateVars), high=np.full(self.nStateVars, 1), dtype=np.float32)
        # observation space variables/elements scaled to be between 0 and 1
        # min = (0,0,0,0)
        # max = (1,1,1,1)
        

    def step(self, action):
        
        if(self.done): # self.time = bookingHorizon = totalTime = 182
            return None

        reward = 0
               
        
        self.action = action 
        fClassOpen = self.actionMatrix[self.action] # action determines whether the fare classes are open or closed
        # e.g. fClassOpen = [1,1,1]

        
        arvTime = self.currentEpisode[self.paxIndex,0]
        while (arvTime==self.time): # pax arrives within this time step
            fclass = int(self.currentEpisode[self.paxIndex,1]) # fclass = {0,1,2}
            cncTime = self.currentEpisode[self.paxIndex,2]
            # Pax will make a booking if the fare class is open
            if (fClassOpen[fclass]==1): # pax buys the ticket
                self.nBookings[fclass] += 1 # fclass = fare class of passenger who booked the ticket = {0,1,2}
                farePaid = self.fClassPrices[fclass]
                
                # Computing immediate reward
                if (not self.computeRewardAtEnd):
                    reward += farePaid

                # Checking if passenger will cancel reservation later on
                if (cncTime > -1):
                    self.cancellations.append((cncTime, fclass, farePaid))
                    # sort on first index (cancellation time)
                    self.cancellations.sort(key= lambda elem: elem[0])
            
            # Move on to next pax
            if(self.paxIndex < self.nTotalPax - 1):
                self.paxIndex += 1
                arvTime = self.currentEpisode[self.paxIndex,0]
            else:
                arvTime = -1 # break while loop
                

        # removing any cancellations that occur during this time step
        while((len(self.cancellations) > 0) and (self.cancellations[0][0] == self.time)):
            fclassCnc = self.cancellations[0][1]
            farePaid = self.cancellations[0][2]
            self.nBookings[fclassCnc] -= 1            
            self.nCancellations[fclassCnc] += 1
            # Computing reward
            if (not self.computeRewardAtEnd):
                reward -= (farePaid - self.cncFee[fclassCnc]) 
            # Removing first element of the stack
            self.cancellations.pop(0)

        self.time = self.time + 1 # time advances one step in the episode
        # Checking if this is the last time step
        if (self.time==self.totalTime): # if self.time == 182
            self.done = True

        
        if (self.done): # terminal state
            
            # computing bumping cost
            self.overbooking = 0
            if(sum(self.nBookings) > self.capacity): # Oversold/overbooked flight
                nPaxToBump = sum(self.nBookings) - self.capacity # number of passengers that need to voluntarily/involuntarily denied boarding
                self.overbooking = nPaxToBump
                # first bump pax who paid the lowest fare (i.e. belongs to the lowest wtpGroup of the lowest fare class)
                # if(nPaxToBump <= self.nBookings[2]): # if(nPaxToBump <= self.nBookings[5]):
                if(nPaxToBump <= self.nBookings[self.nFareClasses-1]): # if(nPaxToBump <= self.nBookings[3]):
                    self.nBookings[self.nFareClasses-1] -= nPaxToBump                    
                    # Assuming the scheduled flight is a domestic one and bumped pax will experience 1 to 2 hour arrival delay
                    reward -= min(self.overbooking_cost_multiplier*self.fClassPrices[self.nFareClasses-1],675/100)*nPaxToBump # DBC - Denied Boarding Compensation
                elif((nPaxToBump-self.nBookings[self.nFareClasses-1]) <= self.nBookings[self.nFareClasses-2]):
                    # first bump all pax of lowest fare class
                    reward -= min(self.overbooking_cost_multiplier*self.fClassPrices[self.nFareClasses-1],675/100)*self.nBookings[self.nFareClasses-1]
                    nPaxToBump -= self.nBookings[self.nFareClasses-1]                                        
                    self.nBookings[self.nFareClasses-1] = 0                    
                    # then bump middle class pax until nPaxOnBoard = capacity
                    reward -= min(self.overbooking_cost_multiplier*self.fClassPrices[self.nFareClasses-2],675/100)*nPaxToBump
                    self.nBookings[self.nFareClasses-2] -= nPaxToBump
                    # nPaxToBump = 0
                else: # elif((nPaxToBump-self.nBookings[self.nFareClasses-1]-self.nBookings[self.nFareClasses-2]) <= self.nBookings[self.nFareClasses-3]):
                    # first bump all pax of lowest fare class 
                    reward -= min(self.overbooking_cost_multiplier*self.fClassPrices[self.nFareClasses-1],675/100)*self.nBookings[self.nFareClasses-1]
                    nPaxToBump -= self.nBookings[self.nFareClasses-1]
                    self.nBookings[self.nFareClasses-1] = 0
                    # then bump all middle class pax
                    reward -= min(self.overbooking_cost_multiplier*self.fClassPrices[self.nFareClasses-2],675/100)*self.nBookings[self.nFareClasses-2]
                    nPaxToBump -= self.nBookings[self.nFareClasses-2]
                    self.nBookings[self.nFareClasses-2] = 0
                    # then bump high fare class pax until nPaxToBump=0
                    reward -= min(self.overbooking_cost_multiplier*self.fClassPrices[self.nFareClasses-3],675/100)*nPaxToBump
                    # nPaxToBump = 0
                    self.nBookings[self.nFareClasses-3] -= nPaxToBump
                           
        self.reward = reward
        
        if(self.biased):
            # self.observation = (self.time, self.nextClass, self.nBookings, 1)
            self.observation = (self.time/182, self.nBookings/100, 1) # self.nCancellations/10, 1)
        else:
            self.observation = (self.time/182, self.nBookings/100) #, self.nCancellations/10)
        return self.observation, reward, self.done, dict()
    

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """

        # load next Data Set (of next flight episode)
        if (self.currentEpisodeIndex == None): # when environment object has just been created
            self.currentEpisodeIndex = 0
        else:
            self.currentEpisodeIndex += 1 # previous episode has ended; load new episode

        if(self.currentEpisodeIndex >= self.nEpisodes):
            print('No more episode data')

        self.currentEpisode = self.trDataPy[self.currentEpisodeIndex] # load episode cell onto self.currentEpisode
        self.max_reward = self.max_reward_list[self.currentEpisodeIndex] # theoretical upper bound of revenue/theoretical optimal revenue
        self.nArrivalsCurrentEps = self.nArrivals_eachEps[self.currentEpisodeIndex] # number of arrivals of each fare class; 1x3 numeric array

        # reset variables
        self.paxIndex = 0
        self.nTotalPax = self.currentEpisode.shape[0] # = nArrivalsCurrentEps[0] + nArrivalsCurrentEps[1] + nArrivalsCurrentEps[2]
        # self.arrivalTime = self.currentEpisode[0,0] # arrival time of 1st passenger
        self.time = 0
        # when reading in data subtract 1 so that zero indexed
        # self.nextClass = int(self.currentEpisode[0, 1] - 1)
        self.nBookings = np.zeros(self.nFareClasses, dtype=int) # self.nBookings = [0 0 0]
        self.nCancellations = np.zeros(self.nFareClasses-1, dtype=int) 

        self.cancellations = [] # empty list

        self.done = False 

        if(self.biased):
            self.observation = (self.time/182, self.nBookings/100, 1) # self.nCancellations/10, 1)
        else:
            self.observation = (self.time/182, self.nBookings/100) # , self.nCancellations/10)

        return self.observation # return s0 (initial state/obs)

    def render(self, mode='human', close=False):
        
        if(self.done):
            print('Max Reward: ' + repr(self.max_reward))
        return print('State: ' + repr(self.observation) + ' Action: ' + repr(self.action) + ' Reward: ' + repr(self.reward)) 

    def close(self):
       

class armProcessor(Processor):
    
    def __init__(self, biased):
        self.biased = biased

    def process_observation(self, observation):
        

        if(self.biased):
            processed_observation = np.concatenate((observation[0], observation[1], observation[2]), axis=None) # processed_observation = ([time, nSeatsSoldByPrices[0] ... nSeatsSoldByPrices[5], nCancellationsByPrices[0] ... nCancellationsByPrices[3], bias=1]) array
        else:
            processed_observation = np.concatenate((observation[0], observation[1]), axis=None) # processed_observation = ([time, nSeatsSoldByPrices[0] ... nSeatsSoldByPrices[5], nCancellationsByPrices[0] ... nCancellationsByPrices[3]]) array
        return processed_observation 

class infoLogger(Callback): 

    def _set_env(self, env):
        self.env = env

    
    def on_train_begin(self, logs={}):
        # initializing log arrays
        self.nBookings = [] # self.fclassSeatsAllocation or self.fclassSeatsOccupancy at the end of each episode        
        self.nCancellations = [] 
        self.episode_reward = []
        self.max_reward = []
        self.rewardPercentage = [] # self.epsRewardPercentage
        self.overbooking = []
        # self.nArrivals = []
        # self.acceptPercentage = []
        self.loadFactor = []
        
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        

    def on_episode_begin(self, episode, logs={}): # callbacks.on_episode_begin(episode)
        # self.nAccepts = 0
        
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        

    def on_step_end(self, epsode_steps, logs={}): # callbacks.on_step_end(episode_step, step_logs)

        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        

    # compute RewardPercentage (as compared to optimalReward) and loadFactor of each episode 
    def on_episode_end(self, episode, logs={}):
        self.nBookings.append(self.env.nBookings) # infoLoggerObj.nBookings = [nBookingsEps0, nBookingsEps1, ...]
        self.nCancellations.append(self.env.nCancellations) 
        self.episode_reward.append(logs['episode_reward']) 
        self.max_reward.append(self.env.max_reward)
        self.overbooking.append(self.env.overbooking)
        # nTotalPaxCurrentEps = sum(self.env.nArrivalsCurrentEps)
        # self.nArrivals.append(nTotalPaxCurrentEps)
        # pdb.set_trace()
        self.rewardPercentage.append(logs['episode_reward']/self.env.max_reward) 
        # self.acceptPercentage.append(self.nAccepts/nTotalPaxCurrentEps) 
        if(self.env.overbooking > 0): 
            self.loadFactor.append(1 + self.env.overbooking/self.env.capacity) 
        else:
            # maxPossiblePassengers = min(nTotalPaxCurrentEps, self.env.capacity)
            self.loadFactor.append(sum(self.env.nBookings)/self.env.capacity)
        

def running_mean(x, N): # rP, rmInterval
    cumsumArr = np.cumsum(np.insert(x, 0, 0))  # np.insert(x, 0, 0) = np.concatenate(([0],x))
    return (cumsumArr[N:] - cumsumArr[:-N]) / N
