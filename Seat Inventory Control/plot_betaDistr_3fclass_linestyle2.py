import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# t = np.arange(0:1/182:1)
# t = np.linspace(0, 1, num=183) # np.arange(0,182+1)/182



alpha_param = np.array([9.45*3, 7.7*2, 3.85*3])
beta_param = np.array([10*3-alpha_param[1-1], 10*2-alpha_param[2-1], 10*3-alpha_param[3-1]])


fClassMeanArrivals = np.array([25, 40, 90])
# the beta distribution is used to get time-varying lambdaPrime values at each time step
# lambdaPrimeValues = area under beta curve in each time step total area = 1 
lambdaPrimeValues = np.zeros((3,182))
lambdaValues = np.zeros((3,182)) # for each fare class, lambda(t) = lambdaPrime(t)*meanNarrivals

for i in range(3): 
    lambdaPrimeValues[i,:] = beta.cdf([x * 1/182 for x in range(1, 182+1)],alpha_param[i],beta_param[i]) - beta.cdf([x * 1/182 for x in range(0, 181+1)],alpha_param[i],beta_param[i]) # area under beta curve at different time steps
    lambdaValues[i,:] = lambdaPrimeValues[i,:]*fClassMeanArrivals[i] # average number of arrivals (Poisson means) at different time steps


lambdaValues = np.append(lambdaValues, np.zeros((3,1)), axis=1)
fareClassLabels = ["High", "Middle", "Low"]


##lineStyle = ['--', ':', '-.']
t = np.arange(0,182+1) # 0:182

        
plt.plot(t,lambdaValues[0,:],label='Fare class H', color='black', linestyle=':', lw=1)
plt.plot(t,lambdaValues[1,:],label='Fare class M', color='c', linestyle='--')
plt.plot(t,lambdaValues[2,:],label='Fare class L', color='lightcoral', linestyle='-')
plt.legend()
plt.xlabel('Time in booking period (day)')#,**tnrfont)
plt.ylabel('Arrival rate')#,**tnrfont)
##plt.show()
plt.savefig('plot_fclassPaxArrival2.pdf')
plt.close()

##plt.plot(t1, nBookingsFclass0_eps, label='Fare class H', color='y', linestyle=':', lw=1.5)
##
##plt.plot(t1, nBookingsFclass1_eps, label='Fare class M', color='c', linestyle='--', lw=1) # forestgreen; limegreen; green; lime
##
##plt.plot(t1, beforeBumping_nBookingsFclass2_eps, label='Fare class L', color='lightcoral', linestyle='-')

 
