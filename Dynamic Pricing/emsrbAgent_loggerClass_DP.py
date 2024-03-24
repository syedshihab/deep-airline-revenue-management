class infoLogger_emsrb: # TODO: Generate graphs showing how the agent varies price of the fare products with time in each episode
       
    def __init__(self):
        # initializing log arrays
        self.nBookings_fp = [] # self.fproductSeatsAllocation or self.fproductSeatsOccupancy at the end of each episode
        self.nBookings_price = []
        self.preBumping_nBookings_fp = []
        self.preBumping_nBookings_price = []
        self.bumpingCost = []
        self.nCancellations_price = [] 
        self.episode_reward = []
        self.max_reward = []
        self.rewardPercentage = [] # self.epsRewardPercentage
        self.overbooking = []
        # self.nArrivals = []
        # self.acceptPercentage = []
        self.loadFactor = []
        
        self.observations = {}
        self.rewards = {}
        self.prices_fp = {}        
        self.openClose_WTPgrpFC = {}
        self.BLemsrb = {}
        

    def on_episode_begin(self, episode): # callbacks.on_episode_begin(episode)
        # self.nAccepts = 0
        
        self.observations[episode] = []
        self.rewards[episode] = []
        self.prices_fp[episode] = []
        self.openClose_WTPgrpFC[episode] = []
        self.BLemsrb[episode] = []
        

    def on_step_end(self, eps, obs, reward, openClose_WTPgrpFC, BLemsrb, prices_fp): # callbacks.on_step_end(episode_step, step_logs)

        # print('testLogger.observations[eps] =',self.observations[eps])
        self.observations[eps].append(obs)
        # print('testLogger.observations[eps] =',self.observations[eps])
        self.rewards[eps].append(reward)
                
        self.openClose_WTPgrpFC[eps].append(openClose_WTPgrpFC)
        self.BLemsrb[eps].append(BLemsrb)
        self.prices_fp[eps].append(prices_fp)
        

    # compute RewardPercentage (as compared to optimalReward) and loadFactor of each episode 
    def on_episode_end(self, nBookings_fp, nBookings_price, preBumping_nBookings_fp, preBumping_nBookings_price, bumpingCost, nCancellations_price, episode_reward, max_reward, overbooking, capacity):
        self.nBookings_fp.append(nBookings_fp) # infoLoggerObj.seats = [seatsEps0, seatsEps1, ...]
        self.nBookings_price.append(nBookings_price)
        self.preBumping_nBookings_fp.append(preBumping_nBookings_fp)
        self.preBumping_nBookings_price.append(preBumping_nBookings_price)
        self.bumpingCost.append(bumpingCost)
        self.nCancellations_price.append(nCancellations_price) 
        self.episode_reward.append(episode_reward) 
        self.max_reward.append(max_reward)
        self.overbooking.append(overbooking)
        # nTotalPaxCurrentEps = sum(nArrivalsCurrentEps)
        # self.nArrivals.append(nTotalPaxCurrentEps)
        
        self.rewardPercentage.append(episode_reward/max_reward) # TODO: compare with EMSRb reward # record bumping cost incurred in each episode; as the agent gets trained, episode bumping cost should decrease
        # self.acceptPercentage.append(self.nAccepts/nTotalPaxCurrentEps) 
        if(overbooking > 0): 
            self.loadFactor.append(1 + overbooking/capacity) 
        else:
            # maxPossiblePassengers = min(nTotalPaxCurrentEps, capacity)
            self.loadFactor.append(sum(nBookings_fp)/capacity)
               
        
        
