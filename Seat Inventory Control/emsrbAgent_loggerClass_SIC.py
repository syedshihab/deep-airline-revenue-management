class infoLogger_emsrb: # TODO: Generate graphs showing how the agent varies price of the fare products with time in each episode
       
    def __init__(self):
        # initializing log arrays
        self.nBookings_fClass = []
        self.nCancellations = [] 
        self.episode_reward = []
        self.max_reward = []
        self.rewardPercentage = []
        self.overbooking = []
        # self.nArrivals = []
        # self.acceptPercentage = []
        self.loadFactor = []

        self.observations = {}
        self.rewards = {}
        self.fClassOpen = {}
        self.BLemsrb = {}
        

    def on_episode_begin(self, episode): # callbacks.on_episode_begin(episode)
        # self.nAccepts = 0
        
        self.observations[episode] = []
        self.rewards[episode] = []
        self.fClassOpen[episode] = []
        self.BLemsrb[episode] = []
        

    def on_step_end(self, eps, obs, reward, fClassOpen, BLemsrb): # callbacks.on_step_end(episode_step, step_logs)

        # print('testLogger.observations[eps] =',self.observations[eps])
        self.observations[eps].append(obs)
        # print('testLogger.observations[eps] =',self.observations[eps])
        self.rewards[eps].append(reward)
        self.fClassOpen[eps].append(fClassOpen)
        self.BLemsrb[eps].append(BLemsrb)
        

    # compute RewardPercentage (as compared to optimalReward) and loadFactor of each episode 
    def on_episode_end(self, nBookings_fClass, nCancellations, episode_reward, max_reward, overbooking, capacity):
        self.nBookings_fClass.append(nBookings_fClass)
        self.nCancellations.append(nCancellations) 
        self.episode_reward.append(episode_reward) 
        self.max_reward.append(max_reward)
        self.overbooking.append(overbooking)
        # nTotalPaxCurrentEps = sum(self.env.nArrivalsCurrentEps)
        # self.nArrivals.append(nTotalPaxCurrentEps)
        # pdb.set_trace()
        self.rewardPercentage.append(episode_reward/max_reward) # TODO: compare with EMSRb reward # record bumping cost incurred in each episode; as the agent gets trained, episode bumping cost should decrease
        # self.acceptPercentage.append(self.nAccepts/nTotalPaxCurrentEps) 
        if(overbooking > 0): 
            self.loadFactor.append(1 + overbooking/capacity) 
        else:
            # maxPossiblePassengers = min(nTotalPaxCurrentEps, self.env.capacity)
            self.loadFactor.append(sum(nBookings_fClass)/capacity)
