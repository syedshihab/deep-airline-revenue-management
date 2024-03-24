import numpy as np
import gym
from gym import spaces
import scipy.io
import pdb

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.utils import plot_model

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor, Env

# implementing the armEnv environment
# TODO: model interval checkpoint DONE
# TODO: normalize neural network inputs/state variables DONE
# TODO: having a bias at each hidden layer

class armEnv(Env): # armEnv is a subclass of the Env abstract class
    """The abstract environment class that is used by all agents. This class has the exact
    same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
    OpenAI Gym environment implementations, this class only defines the abstract methods without any actual
    implementation.
    To implement your own environment, you need to define the following methods:
    - `step` : the process of transitioning to the next state; environment dynamics; state transition  
    - `reset`
    - `render`
    - `close`
    Refer to the [Gym documentation](https://gym.openai.com/docs/#environments).
    """

    # filename - name of the MATLAB file containing the passenger arrival data and market characteristics
    # TODO: make demandTotal = 1.5*flightCapacity
    def __init__(self, filename, biased=1, computeRewardAtEnd=0, ob=2): 
        self.biased = biased
        self.computeRewardAtEnd = computeRewardAtEnd
        self.overbooking_cost_multiplier = ob
        self.reward_range = (-np.inf, np.inf) # immediate reward
        
        mat = scipy.io.loadmat(filename, squeeze_me=True)

        self.priceMatrix = mat['priceMatrix']        
        # self.action_space = spaces.Box( np.array([-1,0,0]), np.array([+1,+1,+1]))  # steer, gas, brake
        # four discrete action variables
        # self.action_space = spaces.Box( np.array([0,0,0,0]), np.array([+1,+2,+2,+2]), dtype=np.int)
        # self.action_space = spaces.Box( np.array([0,0,0]), np.array([+2,+2,+2]), dtype=np.int)
        # set prices of the three fare classes (from any of the 2 fare levels or price points),
        # open/close fare class #new; when p_i = 0, fareClass_i is closed, either for the remaining duration of the episode till flight departure
        self.action_space = spaces.Discrete(self.priceMatrix.shape[0]) # action_space = {0,1,...,17}; action_space.n = 18
        # TODO: generate price graphs - how the prices of the different fare class are varied with time and arrival passengers of different wtpGroups
        
        self.nFareClasses = mat['nFareClasses']
        self.capacity = mat['capacity']
        
        self.totalTime = mat['bookingHorizon'] # do: vary totalTime
        self.nEpisodes = mat['nEpisodes']-1
        self.nWTPgroups = mat['nWTPgroups']
        # self.nFclassPricePoints = mat['nFclassPricePoints']
        # self.trData = mat['trData']
        self.fclassIdentifierPy = mat['fclassIdentifierPy']
        self.wtp = mat['wtp']
        self.cncFee = mat['cncFee']
        self.fareLevels = mat['fareLevels']
        self.trDataPy = mat['trDataPy']
        self.nArrivals = mat['nArrivals']
        self.currentEpisodeIndex = None
        self.currentEpisode = None
        self.nArrivalsCurrentEps = None # self.currentEpisodeNArrivals # TODO: may not need this redundant
        self.max_reward_list = mat['maxReward'] 
        self.max_reward = 0

        self.paxIndex = 0 # passenger index
        self.nTotalPax = 0 # number of passengers in a certain episode = self.currentEpisodeNArrivals
        self.time = 0 # episode current time; or time remaining till flight departure (end of episode)
        # when reading in data subtract 1 so that zero indexed
        # self.nextClass = 0 # state variable: fare/booking class of the just arrived passenger who is requesting booking 
        self.seats = np.zeros(self.nFareClasses, dtype=int) # state variables: s1,s2,s3
        self.nSeatsSoldByPrices = np.zeros(self.nWTPgroups, dtype=int) # nSeatsSoldByPrices = [0 0 0 0 0 0]; nSeatsSoldByPrices[0] = number of seats sold at $400, nSeatsSoldByPrices[1] = number of seats sold at $300, ...
        self.nFareLevels_lowestFclassExcluded = self.nWTPgroups-len(np.where(self.fclassIdentifierPy==(self.nFareClasses-1))[0])
        self.nCancellationsByPrices = np.zeros(self.nFareLevels_lowestFclassExcluded, dtype=int) # nCancellationsByPrices = [0 0 0 0]; cancellation rate of last fare class = 0
        # nCancellationsByPrices[0] = number of cancellations of tickets sold at $400, nSeatsSoldByPrices[1] = number of cancellations of tickets sold at $300
        
        self.cancellations = []

        self.done = False

        # observation (time, classBookingRequest, number of booked passengers of the different fare classes, number of arrivals of the different fare classes, )
##        self.observation_space = spaces.Tuple((spaces.Box(low=0,high=self.totalTime, shape=(1,), dtype=float), spaces.Discrete(self.nFareClasses), spaces.Box(low=np.zeros(self.nFareClasses), high=np.full(self.nFareClasses, 2*self.capacity), dtype=int),
##                                               spaces.Box(low=np.zeros(self.nFareClasses), high=np.full(self.nFareClasses, 200), dtype=int)))

##        # observation (time, classBookingRequest, number of booked passengers of the different fare classes)
##        self.observation_space = spaces.Tuple((spaces.Box(low=0,high=self.totalTime, shape=(1,), dtype=float), spaces.Discrete(self.nFareClasses),
##                                               spaces.Box(low=np.zeros(self.nFareClasses), high=np.full(self.nFareClasses, 2*self.capacity), dtype=int)))

##        # observation (time, number of booked passengers of the different fare classes) time = {0,1,...,182} self.time = 182 means episode has ended
##        self.observation_space = spaces.Tuple((spaces.Discrete(self.totalTime+1), spaces.Box(low=np.zeros(self.nFareClasses), high=np.full(self.nFareClasses, 2*self.capacity), dtype=int)))

        # observation (time, number of tickets sold at different fares, number of cancellations of t1 ... t4) (t, T1 ... T6, c1 ... c4)
        # number of passengers booked in different fare classes can be determined from the number of tickets sold at different fares; e.g. b1 = t1 + t2
        # self.observation_space = spaces.Tuple((spaces.Discrete(self.totalTime+1), spaces.Box(low=np.zeros(self.nWTPgroups), high=np.full(self.nWTPgroups, self.capacity), dtype=int), spaces.Box(low=np.zeros(self.nFareLevels_lowestFclassExcluded), high=np.full(self.nFareLevels_lowestFclassExcluded, 0.5*self.capacity), dtype=int)))
        self.nStateVars = 1 + self.nWTPgroups + self.nFareLevels_lowestFclassExcluded
        self.observation_space = spaces.Box(low=np.zeros(self.nStateVars), high=np.full(self.nStateVars, 1), dtype=np.float32)
        # min = (0,0,0,0,0,0,0)
        # max = (1,1,1,1,1,1,1)

        # TODO: experiment with different state variables: add b1, b2, ... , b6 for the six different WTP groups; number of cancellations in different fare classes
        # TODO: add storage for list of arrival times
        # TODO: vary discount factor (currently set to 0.995) to facilitate training/convergence

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment (action provided by agent's action selection policy).
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action = immediate reward
            done (boolean): Whether the episode has ended (terminal state reached), in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        if(self.done): # self.time = bookingHorizon = totalTime = 182-1 = 181
            return None

        reward = 0
               
        # action = {0,1,...,26} -> 27 possible combinations of fare class prices and fare class availability
        self.action = action 
        prices = self.priceMatrix[self.action] # action determines the prices of the fare classes

        # Checking if there are pax arriving at this time step who are willing to buy tickets at the current prices          
        arvTime = self.currentEpisode[self.paxIndex,0]
        while (arvTime==self.time): # pax arrives within this time step
            wtpGroup = int(self.currentEpisode[self.paxIndex,1]) # wtpGroup = {0,1,2,3,4,5}
            fclass = self.fclassIdentifierPy[wtpGroup] # fclassIdentifierPy = {0,1,2}
            cncTime = self.currentEpisode[self.paxIndex,2]
            # Determining if pax will buy ticket at the current price - depends on pax's wtpGroup, fclass and fclass_price charged
            if (self.wtp[wtpGroup]>=prices[fclass]): # pax buys the ticket
                self.seats[fclass] += 1 # fclass = fare class of passenger who booked the ticket = {0,1,2}
                farePaid = prices[fclass]
                self.nSeatsSoldByPrices[np.where(self.fareLevels == farePaid)[0][0]] += 1 
                # Computing immediate reward
                if (not self.computeRewardAtEnd):
                    reward += farePaid

                # Checking if passenger will cancel reservation later on
                if (cncTime > -1):
                    self.cancellations.append((cncTime, fclass, farePaid))
                    # sort on first index (cancellation time)
                    self.cancellations.sort(key= lambda elem: elem[0])
##                else: # pax will not cancel
##                    priceInd = np.where(self.fareLevels == farePaid)
##                    self.nSeatsSoldByPrices[priceInd[0][0]] += 1 # At the end of the episode, sum(self.seats) = sum(self.nSeatsSoldByPrices) 
            
            # Move on to next pax
            if(self.paxIndex < self.nTotalPax - 1):
                self.paxIndex += 1
                arvTime = self.currentEpisode[self.paxIndex,0]
            else:
                arvTime = -1 # break while loop
                
##                self.time = self.currentEpisode[self.paxIndex, 0]
##                self.nextClass = int(self.currentEpisode[self.paxIndex, 1] - 1)
##            else: # TODO: complete this else section
##                self.done = True # may be when self.time = 182, self.done=True
##                # self.time = self.totalTime 
####                self.nextClass = -1; # TODO: no need for nextClass; remove it

        # removing any cancellations that occur during this time step
        while((len(self.cancellations) > 0) and (self.cancellations[0][0] == self.time)):
            fclassCnc = self.cancellations[0][1]
            farePaid = self.cancellations[0][2]
            self.seats[fclassCnc] -= 1
            fareInd = np.where(self.fareLevels == farePaid)[0][0]
            self.nSeatsSoldByPrices[fareInd] -= 1
            self.nCancellationsByPrices[fareInd] += 1
            # Computing reward
            if (not self.computeRewardAtEnd):
                reward -= (farePaid - self.cncFee[fclassCnc]) # self.fClassCancelRefund[fclassCnc]*self.fclassPrices[fclassCnc]
            # Removing first element of the stack
            self.cancellations.pop(0)

        self.time = self.time + 1 # time advances one step in the episode
        # Checking if this is the last time step
        if (self.time==self.totalTime): # if self.time == 182
            self.done = True
##        else:
##            self.time = self.time + 1 # time advances one step in the episode
        
        
        if (self.done):
            # give reward all at end
##            if self.computeRewardAtEnd:
##                reward = np.dot(self.seats, self.fclassPrices) [b1..b6]*[wtp1...wtp6]
            # computing bumping cost
            self.overbooking = 0
            if(sum(self.seats) > self.capacity): # Oversold/overbooked flight
                nPaxToBump = sum(self.seats) - self.capacity # number of passengers that need to voluntarily/involuntarily denied boarding
                self.overbooking = nPaxToBump
                # first bump pax who paid the lowest fare (i.e. belongs to the lowest wtpGroup of the lowest fare class)
                # if(nPaxToBump <= self.seats[2]): # if(nPaxToBump <= self.seats[5]):
                if(nPaxToBump <= self.nSeatsSoldByPrices[self.nWTPgroups-1]): # if(nPaxToBump <= self.seats[5]):
                    self.seats[self.nFareClasses-1] -= nPaxToBump
                    self.nSeatsSoldByPrices[self.nWTPgroups-1] -= nPaxToBump
                    # Assuming the scheduled flight is a domestic one and bumped pax will experience 1 to 2 hour arrival delay
                    reward -= min(self.overbooking_cost_multiplier*self.fareLevels[self.nWTPgroups-1],675)*nPaxToBump # DBC - Denied Boarding Compensation
                elif((nPaxToBump-self.nSeatsSoldByPrices[self.nWTPgroups-1]) <= self.nSeatsSoldByPrices[self.nWTPgroups-2]):
                    # first bump all pax of lowest wtpGroup
                    reward -= min(self.overbooking_cost_multiplier*self.fareLevels[self.nWTPgroups-1],675)*self.nSeatsSoldByPrices[self.nWTPgroups-1]
                    nPaxToBump -= self.nSeatsSoldByPrices[self.nWTPgroups-1]                                        
                    self.seats[self.nFareClasses-1] -= self.nSeatsSoldByPrices[self.nWTPgroups-1]
                    self.nSeatsSoldByPrices[self.nWTPgroups-1] = 0
                    # then bump middle class pax until nPaxOnBoard = capacity
                    reward -= min(self.overbooking_cost_multiplier*self.fareLevels[self.nWTPgroups-2],675)*nPaxToBump
                    self.seats[self.nFareClasses-1] -= nPaxToBump
                    self.nSeatsSoldByPrices[self.nWTPgroups-2] -= nPaxToBump
                elif((nPaxToBump-self.nSeatsSoldByPrices[5]-self.nSeatsSoldByPrices[4]) <= self.nSeatsSoldByPrices[3]):
                    # first bump all pax of lowest fare class (lowest two wtpGroups)
                    reward -= min(self.overbooking_cost_multiplier*self.fareLevels[5],675)*self.nSeatsSoldByPrices[5]
                    nPaxToBump -= self.nSeatsSoldByPrices[5]
                    self.nSeatsSoldByPrices[5] = 0
                    reward -= min(self.overbooking_cost_multiplier*self.fareLevels[4],675)*self.nSeatsSoldByPrices[4]
                    nPaxToBump -= self.nSeatsSoldByPrices[4]
                    self.nSeatsSoldByPrices[4] = 0    
                    self.seats[2] = 0
                    # then bump middle class pax until nPaxOnBoard = capacity
                    reward -= min(self.overbooking_cost_multiplier*self.fareLevels[3],675)*nPaxToBump
                    self.seats[1] -= nPaxToBump
                    self.nSeatsSoldByPrices[3] -= nPaxToBump
                # TODO: add another elif
           
        self.reward = reward
        
        if(self.biased):
            # self.observation = (self.time, self.nextClass, self.seats, 1)
            self.observation = (self.time/182, self.nSeatsSoldByPrices/100, self.nCancellationsByPrices/10, 1)
        else:
            self.observation = (self.time/182, self.nSeatsSoldByPrices/100, self.nCancellationsByPrices/10)
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

        self.currentEpisode = self.trDataPy[self.currentEpisodeIndex]
        self.max_reward = self.max_reward_list[self.currentEpisodeIndex] # theoretical upper bound of revenue/theoretical optimal revenue
        self.nArrivalsCurrentEps = self.nArrivals[self.currentEpisodeIndex] # number of arrivals of each fare class; 1x3 numeric array

        # reset variables
        self.paxIndex = 0
        self.nTotalPax = self.currentEpisode.shape[0] # = nArrivalsCurrentEps[0] + nArrivalsCurrentEps[1] + nArrivalsCurrentEps[2]
        # self.arrivalTime = self.currentEpisode[0,0] # arrival time of 1st passenger
        self.time = 0
        # when reading in data subtract 1 so that zero indexed
        # self.nextClass = int(self.currentEpisode[0, 1] - 1)
        self.seats = np.zeros(self.nFareClasses, dtype=int) # self.seats = [0 0 0]
        self.nSeatsSoldByPrices = np.zeros(self.nWTPgroups, dtype=int)
        self.nCancellationsByPrices = np.zeros(self.nFareLevels_lowestFclassExcluded, dtype=int) 

        self.cancellations = [] # empty list

        self.done = False 

        if(self.biased):
            self.observation = (self.time/182, self.nSeatsSoldByPrices/100, self.nCancellationsByPrices/10, 1)
        else:
            self.observation = (self.time/182, self.nSeatsSoldByPrices/100, self.nCancellationsByPrices/10)

        return self.observation # return s0 (initial state/obs)

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)
        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        # outfile = StringIO() if mode == 'ansi' else sys.stdout
        # outfile.write('State: ' + repr(self.observation) + ' Action: ' + repr(self.action) + '\n')
##        if(self.action == 0):
##            print('Denied!!')
        if(self.done):
            print('Max Reward: ' + repr(self.max_reward))
        return print('State: ' + repr(self.observation) + ' Action: ' + repr(self.action) + ' Reward: ' + repr(self.reward)) # TODO: include number of arrivals of different wtpGroups in the current step in the visual rendering

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """

class armProcessor(Processor):
    """Abstract base class for implementing processors.
    A processor acts as a coupling mechanism between an `Agent` and its `Env`. This can
    be necessary if your agent has different requirements with respect to the form of the
    observations, actions, and rewards of the environment. By implementing a custom processor,
    you can effectively translate between the two without having to change the underlaying
    implementation of the agent or environment.
    Do not use this abstract base class directly but instead use one of the concrete implementations
    or write your own.
    """
    def __init__(self, biased):
        self.biased = biased

    def process_observation(self, observation):
        """Processes the observation as obtained from the environment for use in an agent and
        returns it.
        # Arguments
            observation (object): An observation as obtained by the environment
        # Returns
            Observation obtained by the environment processed
        """

        if(self.biased):
            processed_observation = np.concatenate((observation[0], observation[1], observation[2], observation[3]), axis=None) # processed_observation = ([time, nSeatsSoldByPrices[0] ... nSeatsSoldByPrices[5], nCancellationsByPrices[0] ... nCancellationsByPrices[3], bias=1]) array
        else:
            processed_observation = np.concatenate((observation[0], observation[1], observation[2]), axis=None) # processed_observation = ([time, nSeatsSoldByPrices[0] ... nSeatsSoldByPrices[5], nCancellationsByPrices[0] ... nCancellationsByPrices[3]]) array
        return processed_observation 

class infoLogger(Callback): # TODO: Generate graphs showing how the agent varies price of the fare products with time in each episode

    def _set_env(self, env):
        self.env = env

    
    def on_train_begin(self, logs={}):
        # initializing log arrays
        self.seats = [] # self.fclassSeatsAllocation or self.fclassSeatsOccupancy at the end of each episode
        self.nSeatsSoldByPrices = [] 
        self.nCancellationsByPrices = [] 
        self.episode_reward = []
        self.max_reward = []
        self.rewardPercentage = [] # self.epsRewardPercentage
        self.overbooking = []
        # self.nArrivals = []
        # self.acceptPercentage = []
        self.loadFactor = []
        # TODO: store bumpingCost incurred in each episode (at the end of each eps)
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        

    def on_episode_begin(self, episode, logs={}): # callbacks.on_episode_begin(episode)
        # self.nAccepts = 0
        
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        

    def on_step_end(self, epsode_steps, logs={}): # callbacks.on_step_end(episode_step, step_logs)
##        if(logs['action'] == 1): # if (self.env.action == )
##            self.nAccepts += 1
        # TODO: record action taken
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        

    # compute RewardPercentage (as compared to optimalReward) and loadFactor of each episode 
    def on_episode_end(self, episode, logs={}):
        self.seats.append(self.env.seats) # infoLoggerObj.seats = [seatsEps0, seatsEps1, ...]
        self.nSeatsSoldByPrices.append(self.env.nSeatsSoldByPrices) 
        self.nCancellationsByPrices.append(self.env.nCancellationsByPrices) 
        self.episode_reward.append(logs['episode_reward']) 
        self.max_reward.append(self.env.max_reward)
        self.overbooking.append(self.env.overbooking)
        # nTotalPaxCurrentEps = sum(self.env.nArrivalsCurrentEps)
        # self.nArrivals.append(nTotalPaxCurrentEps)
        # pdb.set_trace()
        self.rewardPercentage.append(logs['episode_reward']/self.env.max_reward) # TODO: compare with EMSRb reward # record bumping cost incurred in each episode; as the agent gets trained, episode bumping cost should decrease
        # self.acceptPercentage.append(self.nAccepts/nTotalPaxCurrentEps) 
        if(self.env.overbooking > 0): 
            self.loadFactor.append(1 + self.env.overbooking/self.env.capacity) 
        else:
            # maxPossiblePassengers = min(nTotalPaxCurrentEps, self.env.capacity)
            self.loadFactor.append(sum(self.env.seats)/self.env.capacity)
        

def running_mean(x, N): # rP, rmInterval
    cumsumArr = np.cumsum(np.insert(x, 0, 0))  # np.insert(x, 0, 0) = np.concatenate(([0],x))
    return (cumsumArr[N:] - cumsumArr[:-N]) / N
