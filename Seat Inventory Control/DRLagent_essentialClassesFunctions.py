import numpy as np

class infoLogger(): # TODO: Generate graphs showing how the agent varies price of the fare products with time in each episode

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
        self.nBookings.append(self.env.nBookings) # infoLoggerObj.nBookings = [nBookingsEps0, nBookingsEps1, ...]
        self.nCancellations.append(self.env.nCancellations) 
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
            self.loadFactor.append(sum(self.env.nBookings)/self.env.capacity)
        

def running_mean(x, N): # rP, rmInterval
    cumsumArr = np.cumsum(np.insert(x, 0, 0))  # np.insert(x, 0, 0) = np.concatenate(([0],x))
    return (cumsumArr[N:] - cumsumArr[:-N]) / N
