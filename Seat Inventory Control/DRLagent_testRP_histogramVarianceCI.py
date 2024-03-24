# This script is used to create the files DRL_testResults_moreDetails.txt and EMSRb_testResults_moreDetails.txt in the folder numericalExperimentsResults_500
# The files contain information on the following for each numerical experiment: bumpingCostFactor, cancelProbsSetting, meanNarrivalsSetting, mean RP (%),
# std RP (%), mean LF (%), std LF (%), mean revenue ($), std revenue ($), 95% C.I. revenue ($)

import pickle
import numpy as np
import scipy.stats
from DRLagent_armClasses_3fClasses_Tb1b2b3_SIC import infoLogger
from emsrbAgent_loggerClass_SIC import infoLogger_emsrb

# confidence interval function
def mean_confidence_interval(data, confidence=0.95):
    # a = 1.0 * np.array(data)
    a = data
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

# Changing font size of all items (labels, ticks, etc.) in the plots
##font = {'size'   : 12} # 'serif':'Computer Modern Roman',  # Times New Roman
##plt.rc('font', **font)

filepointerDRL = open("numericalExperimentsResults_500" + "/DRL_testResults_moreDetails.txt", "w")
filepointerDRL.write("bumpingCostFactor, cancelProbsSetting, meanNarrivalsSetting, mean RP (%), std RP (%), mean LF (%), std LF (%), mean revenue ($), std revenue ($), 95% C.I. revenue ($)" + "\n")

filepointerEMSRb = open("numericalExperimentsResults_500" + "/EMSRb_testResults_moreDetails.txt", "w")
filepointerEMSRb.write("bumpingCostFactor, cancelProbsSetting, meanNarrivalsSetting, mean RP (%), std RP (%), mean LF (%), std LF (%), mean revenue ($), std revenue ($), 95% C.I. revenue ($)" + "\n")

bcfArray = [2] # [2.0, 2.5]
cncArray = [1,2,3]  # [ 2 3]
fdArray = [1,2,3] # fdArray = [3] # changed from fdArray = [1] 
for bumpingCostFactor in bcfArray:
    for cancelProbsSetting in cncArray:
        for meanNarrivalsSetting in fdArray:
            
            outputDir = "numExp_outputs_" + "OB" + repr(bumpingCostFactor) + "cnc" + repr(cancelProbsSetting) + "_fcArrivals" + repr(meanNarrivalsSetting) + "_" + repr(1) # location where output plots and model weights are saved # TODO: change last digit
                        
            test_logger = infoLogger()

            with open(outputDir + '/testLogger_bestModel_500.pkl','rb') as f:  # Python 3: open(..., 'rb')
                test_logger.nBookings, test_logger.nCancellations, test_logger.episode_reward, test_logger.max_reward, test_logger.rewardPercentage, \
                test_logger.overbooking, test_logger.loadFactor, test_logger.observations, test_logger.rewards, test_logger.actions = pickle.load(f)

            test_logger.rewardPercentage = np.array(test_logger.rewardPercentage)*100
            test_logger.loadFactor = np.array(test_logger.loadFactor)*100
            test_logger.episode_reward = np.array(test_logger.episode_reward)*100

            filepointerDRL.write(repr(bumpingCostFactor) + ", ")
            filepointerDRL.write(repr(cancelProbsSetting) + ", ")
            filepointerDRL.write(repr(meanNarrivalsSetting) + ", ")
            filepointerDRL.write("{:.2f}".format(np.mean(test_logger.rewardPercentage)) + ", ")
            filepointerDRL.write("{:.2f}".format(np.std(test_logger.rewardPercentage)) + ", ")
            filepointerDRL.write("{:.2f}".format(np.mean(test_logger.loadFactor)) + ", ")
            filepointerDRL.write("{:.2f}".format(np.std(test_logger.loadFactor)) + ", ")
            filepointerDRL.write("{:.2f}".format(np.mean(test_logger.episode_reward)) + ", ")
            filepointerDRL.write("{:.2f}".format(np.std(test_logger.episode_reward)) + ", ")
            filepointerDRL.write(u"\u00B1" + "{:.2f}".format(mean_confidence_interval(test_logger.episode_reward)) + "\n")

            f.close()

            # EMSRb
            pklFilename = "emsrbAgent_testLogger_cnc" + repr(cancelProbsSetting) + "_fcArrivals" + repr(meanNarrivalsSetting) + ".pkl"
            test_logger_EMSRb = infoLogger_emsrb()
            
            with open(pklFilename,'rb') as f_EMSRb:  # Python 3: open(..., 'rb')
                test_logger_EMSRb.nBookings_fClass, test_logger_EMSRb.nCancellations, test_logger_EMSRb.episode_reward, test_logger_EMSRb.max_reward, \
                test_logger_EMSRb.rewardPercentage, test_logger_EMSRb.overbooking, test_logger_EMSRb.loadFactor, test_logger_EMSRb.observations, \
                test_logger_EMSRb.rewards, test_logger_EMSRb.fClassOpen, test_logger_EMSRb.BLemsrb = pickle.load(f_EMSRb)

            test_logger_EMSRb.rewardPercentage = np.array(test_logger_EMSRb.rewardPercentage)*100
            test_logger_EMSRb.loadFactor = np.array(test_logger_EMSRb.loadFactor)*100
            test_logger_EMSRb.episode_reward = np.array(test_logger_EMSRb.episode_reward)*100

            filepointerEMSRb.write(repr(bumpingCostFactor) + ", ")
            filepointerEMSRb.write(repr(cancelProbsSetting) + ", ")
            filepointerEMSRb.write(repr(meanNarrivalsSetting) + ", ")
            filepointerEMSRb.write("{:.2f}".format(np.mean(test_logger_EMSRb.rewardPercentage)) + ", ")
            filepointerEMSRb.write("{:.2f}".format(np.std(test_logger_EMSRb.rewardPercentage)) + ", ")
            filepointerEMSRb.write("{:.2f}".format(np.mean(test_logger_EMSRb.loadFactor)) + ", ")
            filepointerEMSRb.write("{:.2f}".format(np.std(test_logger_EMSRb.loadFactor)) + ", ")
            filepointerEMSRb.write("{:.2f}".format(np.mean(test_logger_EMSRb.episode_reward)) + ", ")
            filepointerEMSRb.write("{:.2f}".format(np.std(test_logger_EMSRb.episode_reward)) + ", ")
            filepointerEMSRb.write(u"\u00B1" + "{:.2f}".format(mean_confidence_interval(test_logger_EMSRb.episode_reward)) + "\n")

            f_EMSRb.close()


filepointerDRL.close()
filepointerEMSRb.close()


##            plt.hist(test_logger.episode_reward, bins='auto')
##            plt.show()           

##            plt.hist(test_logger.rewardPercentage, bins='auto')
##            plt.show()

##print(np.std(test_logger.episode_reward,ddof=1))
##print(np.std(test_logger.rewardPercentage,ddof=1))


            
