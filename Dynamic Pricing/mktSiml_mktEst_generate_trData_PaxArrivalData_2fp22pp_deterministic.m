% simulator of the true market simulator based on market estimates

% CHANGED fProductPricePoints

% TODO: after making any changes to nPricePoints/nWTPgroups/fProductPrices in any fare class, change fProductBookingWindow, fProductDistrMeans, fProductCancelRate, arrivalStartTimes_wtpGrps, arrivalEndTimes_wtpGrps
% TODO: after making changes to nFareClasses, change fProductBookingWindow, fProductPrices, price matrix code, fProductDistrMeans, fProductCancelRate , cncFee, arrivalStartTimes_wtpGrps, arrivalEndTimes_wtpGrps

% Reward scaled
% ARM problem with 2 fare classes and booking horizon = DCP = 182
% demands of the two fare classes are stochastic because the inter-arrival times of the passengers of difderent fare classes follows an exponential random distribution
% in other words, number of arrivals of each fare class within each time step follows a Poisson distribution
% cancellations occur based on the fare class cancellation rates
% number of price points of each fare class has been increased

clear all;
close all;

% TODO: new arrival model based on NHPP (polya process) and perhaps new cancellation model also based on MHPP
% Parameters % TODO: use real world problem parameters

load('mktParameterEstimates')

demandDistrType = 'NHPP';
interArrivalDistrType = 'exponential random';

nEpisodes = 50000; % 100000; % 85000; % 50000; % 5000; % 500000; % 400000; % 200000; % 1000; % 200000; % 10000; % 200000; % 25000; % number of episodes used for training the DRL agent
% nEpisodes = 5000; % 50000; % 5000; % 500000; % 400000; % 200000; % 1000; % 200000; % 10000; % 200000; % 25000; % number of episodes used for testing the DRL agent
% fileName = 'trData_CPS1_MAS1_2fc33pp_1_practice'; % 'training_c1_fd1_startTr4_test';

nFareProducts = 2; % low fare product and high fare product
capacity = 100; % flight capacity
bookingHorizon = 182; % 6 months % 365 % 1000; % total booking horizon/window/interval

fProductPricePoints = {[6, 4], [2, 1.5]}; % {[6, 4], [2, 1]}; % fare product prices will be set by agent
% fProductBookingWindow = [7, 7, 7, 21, 42, 119]; 

nfProductPricePoints = cellfun(@length,fProductPricePoints); % = [2, 2]
nWTPgroups = sum(nfProductPricePoints); % = 4


fProductIdentifier = [];
for i=1:nFareProducts
    fProductIdentifier = [fProductIdentifier repelem(i,nfProductPricePoints(i))]; 
end
fProductIdentifierPy = fProductIdentifier - 1; % [0 0 1 1]

wtp = [];
for i=1:nFareProducts
    wtp = [wtp fProductPricePoints{i}];
end
% wtp = [fProductPrices{1} fProductPrices{2}]; 
fareLevels = wtp; % = [600, 400, 200, 100] = cell2flatArray(fProductPricePoints)

% priceMatrix = zeros(prod(nfProductPricePoints+1),nFareClasses);
priceMatrix = zeros(prod([nfProductPricePoints(1) nfProductPricePoints(2:end)+1]),nFareProducts);
rowInd = 1;
% number of loops = nFareClasses
% loop_indexValues = [flip(fProductPrices{1} [flip(fProductPrices{2}) inf]]
for p1=flip(fProductPricePoints{1}) % most expensive fare class will never be closed;
    for p2=[flip(fProductPricePoints{2}) inf] 
        % for p3=[flip(fProductPrices{3}) inf] % 0:2 
            priceMatrix(rowInd,:)=[p1 p2];
            rowInd = rowInd + 1;
        % end
    end
end


% % % % % wtpGroupMeanArrivals_allExp = [14, 21, 40, 70;...
% % % % %                                14, 25, 20, 90;...
% % % % %                                20, 30, 40, 60]; % the mean number of passenger arrivals of the different WTPgroups in three different settings (experiments); 
% % % % % Note that mean totalDemand =~ 1.5*capacity =~ 150 in all experiments


% % % % % % cncRate_fp_last2timeSteps_allExp = [0.05, 0;...
% % % % % %                                     0.1, 0;...
% % % % % %                                     0.15, 0]; % booking/ticket cancellation rates in the last two time steps for each fare product in each setting (experiment); 

cncFee = [0, 0]; % wtpGroupCancelRefund = [4 1 0]; % lowest fare class passengers do not have the option of cancelling their booking
% TODO: set cncFee = [0, 0, 0]; if there is no canellation fee, there is no need for cancellation count variables c1,c2,c3

% Beta distributions represent the nonhomogeneous (time-varying) arrival rates of the different fare classes
% Beta distribution shape parameters for each fare class are given by alpha and beta
% wtpGroup1, wtpGroup2 and wtpGroup3 peak arrivals occur at t=0.945, 0.85, 0.77, 0.65, 0.5 and 0.385 of booking period respectively; so means of beta distributions = 0.945, 0.77 and 0.385 respectively
% % % % % alpha = [9.45*3, 7.7*2, 6.0*2, 4.00*3];  
% % % % % beta = [10*3-alpha(1), 10*2-alpha(2), 10*2-alpha(3), 10*3-alpha(4)];

% alpha = [9.45*3, 7.7*2, 3.85*3];  
% beta = [10*3-alpha(1), 10*2-alpha(2), 10*3-alpha(3)];

% wtpGroup1cancellationProbs = [0.01,...,0.1,0.1];
%         wtpGroup2cancellationProbs = [0.01,...,0.05,0.05];
%         wtpGroup3cancellationProbs = [0,...,0];

% actionMatrix = [1 0 0;... % action=0
%                 1 1 0;... % action=1
%                 1 1 1];   % action=2

                
for meanNarrivalsSetting=1:1
    for cancelProbsSetting=1:1        
                
        tic
        fprintf('meanNarrivalsSetting = %i\n',meanNarrivalsSetting)
        fprintf('cancelProbsSetting = %i\n',cancelProbsSetting)
        
% % % % %         wtpGroupMeanArrivals = wtpGroupMeanArrivals_allExp(meanNarrivalsSetting,:); % mean number of arrivals of passengers in different WTP groups/categories; % totalDemand = 1.5*capacity = 150 if capacity=100 
        % wtpGroupCancelRate = wtpGroupCncRate_last2timeSteps_allExp(cancelProbsSetting,:); % cancellation rates of each fare class;
% % % % %         cncProb_wtpGroups = [repelem(0.0001,1,180), cncRate_fp_last2timeSteps_allExp(cancelProbsSetting,1), cncRate_fp_last2timeSteps_allExp(cancelProbsSetting,1);...
% % % % %                              repelem(0.0001,1,180), cncRate_fp_last2timeSteps_allExp(cancelProbsSetting,1), cncRate_fp_last2timeSteps_allExp(cancelProbsSetting,1)];
                             
        cncProb_wtpGroups = cncProb_wtpGroups_est;
        
        fileName = 'trData_cnc' + string(cancelProbsSetting) + '_fcArrivals' + string(meanNarrivalsSetting) + '_2fp22pp_mktSiml_mktEst_det'; 
        % fileName = 'testData_cnc' + string(cancelProbsSetting) + '_fcArrivals' + string(meanNarrivalsSetting) + '_2fp22pp_mktSiml_mktEst_det'; % to generate test data, comment out the 
%         % previous line and uncomment this line; also, change nEpisodes to the desired number of test episodes (e.g. 1000)
        
        
        % the beta distribution is used to get time-varying lambdaPrime values at each time step
        % lambdaPrimeValues = area under beta curve in each time step; total area = 1; 
% % % % %         lambdaPrimeValues = zeros(nWTPgroups,182);
% % % % %         lambdaValues = zeros(nWTPgroups,182); % for each fare class, lambda(t) = lambdaPrime(t)*meanNarrivals
% % % % %         for i=1:nWTPgroups
% % % % %             lambdaPrimeValues(i,:) = betacdf((1:182)/182,alpha(i),beta(i)) - betacdf((0:181)/182,alpha(i),beta(i)); % area under beta curve at different time steps
% % % % %             lambdaValues(i,:) = lambdaPrimeValues(i,:)*wtpGroupMeanArrivals(i); % average number of arrivals (Poisson means) at different time steps
% % % % %         end
        
        lambdaValues = lambdaValues_est;
        
        % sample from exprnd with 1/lambdaValues to get exact arrival times of passengers of different fare classes in each time step
        % sample from poissrnd with lambdaValues to get exact nArrivals of passengers of different fare classes in each time step
        
        % DETERMINISTIC nArrivals
        % nArrivals(wtpGroup,timeStep,eps) = nArrivals of passengers of wtpGroup in time step timeStep in episode eps
        nArrivals = zeros(nWTPgroups,bookingHorizon,nEpisodes);
        for wtpGroup=1:nWTPgroups
            for time=1:bookingHorizon
                nArrivals(wtpGroup,time,:) = repelem(lambdaValues(wtpGroup,time),1,nEpisodes);
            end
        end
        
        trData = cell(1,nEpisodes);
        for eps=1:nEpisodes
            for time=1:bookingHorizon
                for wtpGroup=1:nWTPgroups
                    trData{eps} = [trData{eps}; repmat([time,wtpGroup,0],nArrivals(wtpGroup,time,eps),1)];
                end
            end
        end
        
        % modeling cancellations and setting cancellation times in trData% 
        % DETERMINISTIC
        nCancellations_by_eps_wtpGroup = zeros(nEpisodes,nWTPgroups);
        nCancellations_by_wtpGroup_time_eps = zeros(nWTPgroups,bookingHorizon,nEpisodes);
        for eps=1:nEpisodes
            rowInd_wtpGroup1 = find(trData{eps}(:,2)==1); % find rows in trData where the WTPgroup of the arrival passenger is 1
            rowInd_wtpGroup2 = find(trData{eps}(:,2)==2); % find rows in trData where the WTPgroup of the arrival passenger is 2
%             rowInd_wtpGroup3 = find(trData{eps}(:,2)==3); % find rows in trData where the WTPgroup of the arrival passenger is 3
            
            for ind1=1:length(rowInd_wtpGroup1) % looping through all passengers of WTPgroup 1
                % determining the cancellation time of passenger ind1 of wtpGroup1; if the passenger does not cancel, cancellationTime=0
                arrivalTime = trData{eps}(rowInd_wtpGroup1(ind1),1);
                if arrivalTime<182 % assumption: passengers do not cancel in the time step in which they made a booking
                    for timeInd=(arrivalTime+1):182
                        willCancel = binornd(1,cncProb_wtpGroups(1,timeInd)); % number of trials=1; passenger cancellation decision depends on cancellation probability
                        if willCancel==1
                            trData{eps}(rowInd_wtpGroup1(ind1),3) = timeInd; % setting cancellation time in trData
                            nCancellations_by_eps_wtpGroup(eps,1) = nCancellations_by_eps_wtpGroup(eps,1) + 1;
                            nCancellations_by_wtpGroup_time_eps(1,timeInd,eps) = nCancellations_by_wtpGroup_time_eps(1,timeInd,eps) + 1;
                            break;
                        end
                    end
                end
            end
            
            for ind2=1:length(rowInd_wtpGroup2) % looping through all passengers of fare class 2
                % determining the cancellation time of passenger ind2 of wtpGroup2; if the passenger does not cancel, cancellationTime=0
                arrivalTime = trData{eps}(rowInd_wtpGroup2(ind2),1);
                if arrivalTime<182 % assumption: passengers do not cancel in the time step in which they made a booking
                    for timeInd=(arrivalTime+1):182
                        willCancel = binornd(1,cncProb_wtpGroups(2,timeInd)); % number of trials=1; passenger cancellation decision depends on cancellation probability
                        if willCancel==1
                            trData{eps}(rowInd_wtpGroup2(ind2),3) = timeInd; % setting cancellation time in trData
                            nCancellations_by_eps_wtpGroup(eps,2) = nCancellations_by_eps_wtpGroup(eps,2) + 1;
                            nCancellations_by_wtpGroup_time_eps(2,timeInd,eps) = nCancellations_by_wtpGroup_time_eps(2,timeInd,eps) + 1;
                            break; % no need to check for occurence of cancellation in other time steps
                        end
                    end
                end
            end
                        
            
        end
        
        trDataPy = cellfun(@(c) c-1, trData, 'UniformOutput', false);  
        nArrivals_eachEps = cellfun(@length, trDataPy);
        
        nArrivals_by_eps_wtpGroup_1 = sum(nArrivals,2);
        nArrivals_by_eps_wtpGroup = zeros(nEpisodes,nWTPgroups);
        for eps=1:nEpisodes
            nArrivals_by_eps_wtpGroup(eps,:) = nArrivals_by_eps_wtpGroup_1(:,1,eps)';
        end
               
        
        % nCancellations = cellfun(@(c) sum(ceil(c)), cancellations); % size = 1x1 cell -> 25000 x 3 array 
        % Computing max reward for each episode by considering WTP of committed passengers
        % maxReward(flightEpsIndex) = theoretical upper bound of revenue (maximum possible revenue) in that episode
        % nArrivals - nCancellations = number of confirmed passengers
        optimalNumberBookings = min(nArrivals_by_eps_wtpGroup - nCancellations_by_eps_wtpGroup, capacity*ones(nEpisodes, nWTPgroups)); 
        for i = 2:nWTPgroups
            seatsRemaining = capacity - sum(optimalNumberBookings(:,1:i-1),2);
            optimalNumberBookings(:,i) = min(optimalNumberBookings(:,i), seatsRemaining);
        end
        
        maxReward = optimalNumberBookings*wtp'; % 25000x1 vector
        
        save(strcat(fileName, '.mat'),'-v7');
        
        toc
    end
end
