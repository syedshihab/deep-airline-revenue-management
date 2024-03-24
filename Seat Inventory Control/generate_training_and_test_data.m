% This MATLAB file is used to generate airline passenger arrival and cancellation data for training and testing of AI agents developed to solve the problem of ARM
% Author: Syed A.M. Shihab

% ARM problem with 3 fare classes and booking horizon = nDCPs = 182 days
% Passenger arrival process is modeled as a non-homogeneous Poisson process
% Cancellations occur based on the fare class cancellation rates


clear all;
close all;


% ARM problem parameters

nEpisodes = 100000; % number of episodes used for training the DRL agent
% nEpisodes = 300; 1000; % to generate test data, comment out the previous line and uncomment this line;

nFareClasses = 3;
capacity = 100; % flight capacity
bookingHorizon = 182; % 6 months % total booking horizon/window/period

fClassPrices = [4 2 1]; % Rewards (fare class prices) have been scaled to be between 0 and 10; Original prices: $400, $200 and $100

fclassIdentifier = [1 2 3]; % fare class indices; fare class 1 = high fare class, fare class 2 = middle fare class, fare class 3 = low fare class 
fclassIdentifierPy = fclassIdentifier - 1; % [0 1 2] % fare class indices for use in Python

% the mean number of passenger arrivals of each fare classes in three different settings for carrying out numerical experiments
meanArrivalSetting = [16, 40, 100;... 
                      25, 40, 90;...
                      25, 50, 80]; 

% Note that mean totalDemand =~ 1.5*capacity =~ 150 in all experiments

% passenger booking cancellation rates in the last two time steps for each fare class in three different settings for carrying out numerical experiments
cancellationProbabilitySetting = [0.05, 0.01, 0;...
                                  0.1, 0.05, 0;...
                                  0.15, 0.10, 0]; 

cncFee = [0, 0]; 


% Beta distributions represent the nonhomogeneous (time-varying) arrival rates of the different fare classes
% Beta distribution shape parameters for each fare class are given by alpha and beta
% fClass1, fClass2 and fClass13 peak arrivals occur at t=0.945, 0.77 and 0.385 respectively; so means of beta distributions = 0.945, 0.77 and 0.385 respectively
alpha = [9.45*3, 7.7*2, 3.85*3];  
beta = [10*3-alpha(1), 10*2-alpha(2), 10*3-alpha(3)];


actionMatrix = [1 0 0;... % action=0
                1 1 0;... % action=1
                1 1 1];   % action=2
                
for meanNarrivalsSetting=1:3
    for cancelProbsSetting=1:3        
                
        tic
        fprintf('meanNarrivalsSetting = %i\n',meanNarrivalsSetting)
        fprintf('cancelProbsSetting = %i\n',cancelProbsSetting)
        
        fClassMeanArrivals = meanArrivalSetting(meanNarrivalsSetting,:); % mean number of arrivals of passengers in different WTP groups/categories; % totalDemand = 1.5*capacity = 150 if capacity=100 
        % fClassCancelRate = cancellationProbabilitySetting(cancelProbsSetting,:); % cancellation rates of each fare class;
        cncProb_fClasses = [repelem(0.0001,1,180), cancellationProbabilitySetting(cancelProbsSetting,1), cancellationProbabilitySetting(cancelProbsSetting,1);...
                            repelem(0.0001,1,180), cancellationProbabilitySetting(cancelProbsSetting,2), cancellationProbabilitySetting(cancelProbsSetting,2)];
        
        % fileName = 'trainingData_cnc' + string(cancelProbsSetting) + '_fcArrivals' + string(meanNarrivalsSetting); % to generate training data
        fileName = 'testData_cnc' + string(cancelProbsSetting) + '_fcArrivals' + string(meanNarrivalsSetting); % to generate test data, comment out the 
        % previous line and uncomment this line; also, change nEpisodes to the desired number of test episodes (e.g. 1000)
        
        
        % the beta distribution is used to get time-varying lambdaPrime values at each time step
        % lambdaPrimeValues = area under beta curve in each time step; total area = 1; 
        lambdaPrimeValues = zeros(3,182);
        lambdaValues = zeros(3,182); % for each fare class, lambda(t) = lambdaPrime(t)*meanNarrivals
        for i=1:nFareClasses
            lambdaPrimeValues(i,:) = betacdf((1:182)/182,alpha(i),beta(i)) - betacdf((0:181)/182,alpha(i),beta(i)); % area under beta curve at different time steps
            lambdaValues(i,:) = lambdaPrimeValues(i,:)*fClassMeanArrivals(i); % average number of arrivals (Poisson means) at different time steps
        end

        % sample from exprnd with 1/lambdaValues to get exact arrival times of passengers of different fare classes in each time step
        % sample from poissrnd with lambdaValues to get exact nArrivals of passengers of different fare classes in each time step
        
        % sampling nArrivals from poisson distributions
        % nArrivals(eps,fClass,timeStep) = nArrivals of passengers of fClass in time step timeStep in episode eps
        nArrivals = zeros(nFareClasses,bookingHorizon,nEpisodes);
        for fClass=1:nFareClasses
            for time=1:bookingHorizon
                nArrivals(fClass,time,:) = poissrnd(lambdaValues(fClass,time),1,nEpisodes);
            end
        end
        
        trData = cell(1,nEpisodes);
        for eps=1:nEpisodes
            for time=1:bookingHorizon
                for fClass=1:nFareClasses
                    trData{eps} = [trData{eps}; repmat([time,fClass,0],nArrivals(fClass,time,eps),1)];
                end
            end
        end
        
        % modeling cancellations and setting cancellation times in trData% 
        nCancellations_by_eps_fClass = zeros(nEpisodes,nFareClasses);        
        for eps=1:nEpisodes
            rowInd_fClass1 = find(trData{eps}(:,2)==1); % find rows in trData where the fare class of the arrival passenger is 1
            rowInd_fClass2 = find(trData{eps}(:,2)==2); % find rows in trData where the fare class of the arrival passenger is 2
            
            for ind1=1:length(rowInd_fClass1) % looping through all passengers of fare class 1
                % determining the cancellation time of passenger ind1 of fClass1; if the passenger does not cancel, cancellationTime=0
                arrivalTime = trData{eps}(rowInd_fClass1(ind1),1);
                if arrivalTime<182 % assumption: passengers do not cancel in the time step in which they made a booking
                    for timeInd=(arrivalTime+1):182
                        willCancel = binornd(1,cncProb_fClasses(1,timeInd)); % number of trials=1; passenger cancellation decision depends on cancellation probability
                        if willCancel==1
                            trData{eps}(rowInd_fClass1(ind1),3) = timeInd; % setting cancellation time in trData
                            nCancellations_by_eps_fClass(eps,1) = nCancellations_by_eps_fClass(eps,1) + 1;
                            break;
                        end
                    end
                end
            end
            
            for ind2=1:length(rowInd_fClass2) % looping through all passengers of fare class 2
                % determining the cancellation time of passenger ind1 of fClass2; if the passenger does not cancel, cancellationTime=0
                arrivalTime = trData{eps}(rowInd_fClass2(ind2),1);
                if arrivalTime<182 % assumption: passengers do not cancel in the time step in which they made a booking
                    for timeInd=(arrivalTime+1):182
                        willCancel = binornd(1,cncProb_fClasses(2,timeInd)); % number of trials=1; passenger cancellation decision depends on cancellation probability
                        if willCancel==1
                            trData{eps}(rowInd_fClass2(ind2),3) = timeInd; % setting cancellation time in trData
                            nCancellations_by_eps_fClass(eps,2) = nCancellations_by_eps_fClass(eps,2) + 1;
                            break; % no need to check for occurence of cancellation in other time steps
                        end
                    end
                end
            end
            
        end
        
        trDataPy = cellfun(@(c) c-1, trData, 'UniformOutput', false);  
        nArrivals_eachEps = cellfun(@length, trDataPy);
        
        nArrivals_by_eps_fClass1 = sum(nArrivals,2);
        nArrivals_by_eps_fClass = zeros(nEpisodes,nFareClasses);
        for eps=1:nEpisodes
            nArrivals_by_eps_fClass(eps,:) = nArrivals_by_eps_fClass1(:,1,eps)';
        end
               
        
        % nCancellations = cellfun(@(c) sum(ceil(c)), cancellations); % size = 1x1 cell -> 25000 x 3 array 
        % Computing max reward for each episode by considering WTP of committed passengers
        % maxReward(flightEpsIndex) = theoretical upper bound of revenue (maximum possible revenue) in that episode
        % nArrivals - nCancellations = number of confirmed passengers
        optimalNumberBookings = min(nArrivals_by_eps_fClass - nCancellations_by_eps_fClass, capacity*ones(nEpisodes, nFareClasses)); 
        for i = 2:nFareClasses
            seatsRemaining = capacity - sum(optimalNumberBookings(:,1:i-1),2);
            optimalNumberBookings(:,i) = min(optimalNumberBookings(:,i), seatsRemaining);
        end
        
        maxReward = optimalNumberBookings*fClassPrices'; % 25000x1 vector
        
        save(strcat(fileName, '.mat'),'-v7');
        
        toc
    end
end
