% Reward scaled
% ARM problem with 2 fare classes and booking horizon = 182
% demands of the two fare classes are stochastic
% cancellations occur
% Possibly, training_c1_fd1_startTr6, training_c1_fd1_startTr66, training_c1_fd1_startTr7, and training_c1_fd1_startTr7_bigData
% only vary in the number of training episodes


clear all;
close all;

% TODO: new arrival model based on NHPP (polya process) and perhaps new cancellation model also based on MHPP
% TODO: make fClassPrices = {[600, 400], [300, 200], [150, 100]}; % price will be set by agent

% Parameters % TODO: use real world problem parameters
% nFareClasses = 3; % high, medium and low fare class; each class has two distinct price points associated with it

nFareClasses = 2;
capacity = 100; % 80 % flight capacity; vary
bookingHorizon = 182; % 6 months % 365 % 1000; % total booking horizon/window/interval
% nFclassPricePoints = [2 2 2];

% fClassBookingWindow = [repelem(14,fClassnPricePoints(1)) repelem(28,fClassnPricePoints(2)) repelem(182,fClassnPricePoints(3))]; % TODO: vary
% fClassBookingWindow = [14 28 182];
fClassBookingWindow = [7 7 82 100]; 
nEpisodes = 50000; % 500000; % 400000; % 200000; % 1000; % 200000; % 10000; % 200000; % 25000; % number of episodes used for training the DRL agent

fClassPrices = {[6, 4], [1.5, 1]}; % fare prices will be set by agent
nFclassPricePoints = cellfun(@length,fClassPrices);
nWTPgroups = sum(nFclassPricePoints); % = 6
fclassIdentifier = [];
for i=1:nFareClasses
    fclassIdentifier = [fclassIdentifier repelem(i,nFclassPricePoints(i))]; % TODO: vary; # [1 1 2 2 3 3];
end

fclassIdentifierPy = fclassIdentifier - 1; % [0 0 1 1 2 2]
wtp = [fClassPrices{1} fClassPrices{2}]; 
fareLevels = wtp; % = [400, 300, 250, 200, 125, 100]

% priceMatrix = zeros(prod(nFclassPricePoints+1),nFareClasses);
priceMatrix = zeros(prod([nFclassPricePoints(1) nFclassPricePoints(2:end)+1]),nFareClasses);
rowInd = 1;
for p1=flip(fClassPrices{1}) % [300 400] % most expensive fare class will never be closed; TODO: remove inf; before doing so, test if agent learns this rule
    for p2=[flip(fClassPrices{2}) inf] % 0:2
        % for p3=[flip(fClassPrices{3}) inf] % 0:2 
            priceMatrix(rowInd,:)=[p1 p2];
            rowInd = rowInd + 1;
        % end
    end
end

% fClassPrices = wtp;
% meanDemandL = -0.8*fareL + 140; 
% fClassDistrMeans = [10, 6, 30, 20, 80, 60, 40; 60, 55, 30, 20, 10, 4, 2; 33, 30, 33, 25, 34, 20, 10]; % three different settings of fare class passenger distributions 
% fClassDistrMeans = [10, 30, 60; 60, 30, 10; 33, 33, 34]; % three different settings of fare class passenger distributions 
fClassDistrMeans = [14, 21, 40, 70]; % mean number of arrivals of passengers in different WTP groups/categories; % totalDemand = 1.5*capacity = 150 if capacity=100 
% -> mean number of passenger arrivals of each fare class in each flight episode in three different scenarios/cases
% In row 1, mean(high class pax arrival) = 10, mean(middle class pax arrival) = 30 and mean(low class pax arrival) = 60 

% classCancelRateArray = [0, 0, 0; 0.1, 0.1, 0.1; 0.2, 0.2, 0.2]; % pax cancellation rates of the three fare classes in three different settings
% for fd=1:3
%     for ccr=1:3
%         fareClassSizeMean = fareClassDistributionMeans(fd,:);
%         fClassCancelRate = fClassCancelRateArray(ccr,:)
%         fileName = 'training_cancellations' + string(ccr) + 'fd' + string(fd);
        
fd = 1; % fd = {1,2,3} fd gives the fare class passenger distribution setting 
fClassMeanDemand = fClassDistrMeans(fd,:); % here fclass = WTP mean demand

% wtpGroupCancelRate = fClassCancelRate;
fClassCancelRate = [0.1, 0.1, 0, 0]; % , 0, 0]; % booking/ticket cancellation rates are different for each fare class;
cncFee = [0, 1.5]; % fClassCancelRefund = [1 0.5 0];
% fClassCancelRate = [0.1, 0.05, 0]; 
% lowest fare class passengers do not have the option of cancelling their booking

fileName = 'training_c1_fd1_startTr6c'; % 'training_c1_fd1_startTr4_test';

% sample sizes
% classSizeSD = [5, 5, 10]; % standard deviations of the fare class distributions 
% classSizes(flight episode Index, Fare Class Index)
% Modeling passenger arrival as a Poisson process
% normrnd(fClassMeanDemand(j), classSizeSD(j), 1, nEpisodes)' -> 25000x1 vector of nPassengerArrivals in 25000 flight episodes
% arrayfun(@(j) normrnd(fClassMeanDemand(j), classSizeSD(j), 1, nEpisodes)', 1:nFareClasses, 'UniformOutput', false) -> 1x3 cell, 
% cell{j} = 25000x1 vector of nPassengerArrivals in 25000 flight episodes in fare class j
% classSizes = cell2mat(arrayfun(@(j) normrnd(fClassMeanDemand(j), classSizeSD(j), 1, nEpisodes)', 1:nFareClasses, 'UniformOutput', false));
% classSizes -> 25000x3 matrix of nPassengerArrivals in 25000 flight episodes in all fare classes
% if negative generated force to 0
% classSizes(classSizes<0) = 0;

% Assuming that passengers arrive following a Poisson process with mean number of arrivals given by fClassMeanDemand
avgInterArrivalTime = fClassBookingWindow./fClassMeanDemand; % [14 21 182]./[10 30 60] size = 1 x 3
nArrivalsCellArray = arrayfun(@(lambda)poissrnd(fClassMeanDemand(lambda),nEpisodes,1),1:nWTPgroups,'UniformOutput', false); % size = 1 x 3 cell array
% nArrivalsCellArray = {repelem(fClassDistrMeans(1),nEpisodes,1) repelem(fClassDistrMeans(2),nEpisodes,1) repelem(fClassDistrMeans(3),nEpisodes,1)...
%                       repelem(fClassDistrMeans(4),nEpisodes,1)};

% Assuming that fc_arrival times are approximately uniformly distributed within the respective arrival windows of the fare class
% arrivalTimes_cellRow = {repmat([175:181],1,2)+1 repmat([168:174],1,3)+1 [102:2:180]+1 [0:2:98 repelem(98,10)]+1}; 
% arrivalTimes = repmat(arrivalTimes_cellRow,nEpisodes,1);

% fClassBookingWindow = [7 7 82 100];
arrivalStartTimes_wtpGrps = [176 169 101 1]; % [175 168 168 154 100 0]+1;
arrivalEndTimes_wtpGrps = [182 175 182 100];
arrivalTimes = cell(nEpisodes,nWTPgroups);
for wtpGrp=1:nWTPgroups
    for epsInd=1:nEpisodes
       arrivalTimes{epsInd,wtpGrp} = randi([arrivalStartTimes_wtpGrps(wtpGrp),arrivalEndTimes_wtpGrps(wtpGrp)],1,nArrivalsCellArray{wtpGrp}(epsInd));        
    end
end

% fClassDistrMeans = [8, 12, 15, 20, 40, 60];
% repelem(14,fClassnPricePoints(1))

classCancellations = @(fclass) arrayfun(@(eps) binornd(1, fClassCancelRate(fclass), 1, nArrivalsCellArray{fclass}(eps)).*rand(1, nArrivalsCellArray{fclass}(eps)), 1:nEpisodes, 'UniformOutput', false)';
cancellations = arrayfun(@(fclass) classCancellations(fclass), 1:nWTPgroups, 'UniformOutput', false);
cancellations = horzcat(cancellations{:});


% nCancellations(flight episode Index, Fare Class Index)
nCancellations = cellfun(@(c) sum(ceil(c)), cancellations); % size = 1x1 cell -> 25000 x 3 array 

% Computing max reward for each episode by considering WTP of committed passengers
% maxReward(flightEpsIndex) = theoretical upper bound of revenue (maximum possible revenue) in that episode
% nArrivals - nCancellations = number of confirmed passengers
nArrivals = cell2mat(nArrivalsCellArray); % 25000x6 vector
optimalNumberBookings = min(nArrivals - nCancellations, capacity*ones(size(nArrivals))); 
for i = 2:nWTPgroups
    seatsRemaining = capacity - sum(optimalNumberBookings(:,1:i-1),2);
    optimalNumberBookings(:,i) = min(optimalNumberBookings(:,i), seatsRemaining);
end
maxReward = optimalNumberBookings*wtp'; % 25000x1 vector

% averageMaxReward = mean(maxReward);

% dataSets{i} = [t, fareClassIndex, timeOfCancellation];
% [1, 1, 0] - adding passenger of class 1 at t = 1
% [2, 3, 10] - adding passenger of class 3 at t = 2, will cancel at some later time
% ...

% episode 'j', fare class 'i'           (bookingHorizon-arrivalTimes{j,i}).*cancellations{j,i} = time remaining*cancellation = cancellation time after (wrt) arrival time    
computeCancellationTimeFunc = @(j, i) ((bookingHorizon-arrivalTimes{j,i}).*cancellations{j,i} + arrivalTimes{j,i}.*ceil(cancellations{j,i}))'; % nArrivals(given episode and fare class)x1
compileTrDataFunc = @(j) cell2mat(arrayfun(@(i) [arrivalTimes{j, i}', i*ones(nArrivals(j, i), 1), computeCancellationTimeFunc(j, i)], 1:nWTPgroups, 'UniformOutput', false)');
trData = arrayfun(@(j) compileTrDataFunc(j), 1:nEpisodes, 'UniformOutput', false); % 1x25000 cell array

% sort by arrival times in ascending order
trData = cellfun(@sortrows, trData, 'UniformOutput', false);
trData = cellfun(@(c) c-1, trData, 'UniformOutput', false);

totalNarrivals = sum(nArrivals,2);

trDataPy = cellfun(@(c) ceil(c), trData, 'UniformOutput', false); 
for eps=1:nEpisodes
    trDataPy{eps}((trDataPy{eps}(:,1)<0),1)=0; % practiceArr(practoceArr>10)=10
end
  
% save all variables in the workspace
% save(strcat(fileName, '.mat')) % -> filename.mat

% save(strcat(fileName, '.mat'),'-v7.3');
save(strcat(fileName, '.mat'),'-v7');

% save(strcat(fileName, '.mat'),'-v7.3','-nocompression');
% load(strcat(fileName, '.mat'))
