% TODO: after making any changes to nPricePoints/nWTPgroups/fClassPrices in any fare class, change fClassBookingWindow, fClassDistrMeans, fClassCancelRate, arrivalStartTimes_wtpGrps, arrivalEndTimes_wtpGrps
% TODO: after making changes to nFareClasses, change fClassBookingWindow, fClassPrices, price matrix code, fClassDistrMeans, fClassCancelRate , cncFee, arrivalStartTimes_wtpGrps, arrivalEndTimes_wtpGrps

% Reward scaled
% ARM problem with 2 fare classes and booking horizon = DCP = 182
% demands of the two fare classes are stochastic because the inter-arrival times of the passengers of difderent fare classes follows an exponential random distribution
% cancellations occur based on the fare class cancellation rates
% number of price points of each fare class has been increased

clear all;
close all;

% TODO: new arrival model based on NHPP (polya process) and perhaps new cancellation model also based on MHPP
% Parameters % TODO: use real world problem parameters


demandDistrType = 'stochastic -> Poisson process';
interArrivalDistrType = 'exponential random';

nEpisodes = 5000; % 500000; % 400000; % 200000; % 1000; % 200000; % 10000; % 200000; % 25000; % number of episodes used for training the DRL agent
fileName = 'training_c1_fd1_2fc33pp_1_prc'; % 'training_c1_fd1_startTr4_test';

nFareClasses = 2;
capacity = 100; % 80 % flight capacity; vary
bookingHorizon = 182; % 6 months % 365 % 1000; % total booking horizon/window/interval

fClassPrices = {[6, 5, 4], [2, 1.5, 1]}; % fare prices will be set by agent
fClassBookingWindow = [7, 7, 7, 21, 42, 119]; 

nFclassPricePoints = cellfun(@length,fClassPrices); % = [3, 4]
nWTPgroups = sum(nFclassPricePoints); % = 7

fclassIdentifier = [];
for i=1:nFareClasses
    fclassIdentifier = [fclassIdentifier repelem(i,nFclassPricePoints(i))]; 
end
fclassIdentifierPy = fclassIdentifier - 1; % [0 0 0 1 1 1]

wtp = [];
for i=1:nFareClasses
    wtp = [wtp fClassPrices{i}];
end
% wtp = [fClassPrices{1} fClassPrices{2}]; 
fareLevels = wtp; % = [400, 300, 250, 200, 125, 100]

% priceMatrix = zeros(prod(nFclassPricePoints+1),nFareClasses);
priceMatrix = zeros(prod([nFclassPricePoints(1) nFclassPricePoints(2:end)+1]),nFareClasses);
rowInd = 1;
% number of loops = nFareClasses
% loop_indexValues = [flip(fClassPrices{1} [flip(fClassPrices{2}) inf]]
for p1=flip(fClassPrices{1}) % most expensive fare class will never be closed;
    for p2=[flip(fClassPrices{2}) inf] 
        % for p3=[flip(fClassPrices{3}) inf] % 0:2 
            priceMatrix(rowInd,:)=[p1 p2];
            rowInd = rowInd + 1;
        % end
    end
end

fClassDistrMeans = [8, 8, 20, 15, 20, 70]; % mean number of arrivals of passengers in different WTP groups/categories; % totalDemand = 1.5*capacity = 150 if capacity=100 


fd = 1; % fd = {1,2,3} fd gives the fare class passenger distribution setting 
fClassMeanDemand = fClassDistrMeans(fd,:); % here fclass = WTP mean demand

% wtpGroupCancelRate = fClassCancelRate;
fClassCancelRate = [0.1, 0.1, 0.1, 0, 0, 0]; % , 0, 0]; % booking/ticket cancellation rates are different for each fare class;
cncFee = [0, 1.5]; % fClassCancelRefund = [1 0.5 0]; % lowest fare class passengers do not have the option of cancelling their booking

% Assuming that passengers arrive following a Poisson process with mean number of arrivals given by fClassMeanDemand
avgInterArrivalTime = fClassBookingWindow./fClassMeanDemand; % [14 21 182]./[10 30 60] size = 1 x 3
% passenger inter-arrival times of each wtpGroup follows an exponential random distribution

fClassMaxNarrivals = poissinv(0.9999, fClassDistrMeans); % = poissinv(1, classSizeMean); size: 1 x 6
arrivalStartTimes_wtpGrps = [176, 169, 162, 162, 120, 1];
arrivalEndTimes_wtpGrps =   [182, 175, 168, 182, 161, 119];
% get cell array of arrival times for a single fare class
% using a cell array is necessary to allow for possibly different number of arrivals at each flight/episode
listArrivalTimesFunc = @(i) num2cell(arrivalStartTimes_wtpGrps(i)-1 + cumsum(exprnd(avgInterArrivalTime(i), nEpisodes, fClassMaxNarrivals(i)), 2), 2); % -> exprnd() size = 25000 x maxNumArrivals 
% listArrivalTimes is a function handle of an anonymous function; 'i' is the input argument; num2cell gives a 25000 x 1 cell array in 'listArrivalTimes' 
% for each input fare class 'i'; listArrivalTimes(r) = cell(r) = numeric array of arrivalTimes of passengers for flight 'r'

% flight episode index = episode number or flight number 
% fareClassIndex = {1,2,3}
% arrivalTimes{flight episode Index, Fare Class Index}(arrival Index), arrival Index < nArrivals
arrivalTimes = arrayfun(@(i) listArrivalTimesFunc(i) , 1:nWTPgroups, 'UniformOutput', false); % arrivalTimes is a 1xnWTPgroups cell array
% 'UniformOutput' = false, so that arrayfun returns the outputs of listArrivalTimes in cell arrays. The outputs of listArrivalTimes are of size 25000 x 1
 
% arrivalTimes(1) = cell(1) = listArrivalTimes(1) = 25000 x 1 cell array; 
% listArrivalTimes(r) = cell(r) = numeric array of arrivalTimes of passengers for flight 'r'
% horizontally concatenate the cells of arrivalTimes
arrivalTimes = horzcat(arrivalTimes{:}); % arrivalTimes = 25000x3 cell array; arrivalTimes{2, 3}(5) = arrival time of fifth passenger for third fare class in flight/episode = 2 

% removing arrival times beyond arrivalEndTimes_wtpGrps
for row=1:nEpisodes
    for col=1:nWTPgroups
        arr = arrivalTimes{row,col};
        arrivalTimes{row,col} = arr(arr < arrivalEndTimes_wtpGrps(col));
    end
end

nArrivals = cellfun(@length, arrivalTimes);

% Assuming that fc_arrival times are approximately uniformly distributed within the respective arrival windows of the fare class
% arrivalTimes_cellRow = {repmat([175:181],1,2)+1 repmat([168:174],1,3)+1 [102:2:180]+1 [0:2:98 repelem(98,10)]+1}; 
% arrivalTimes = repmat(arrivalTimes_cellRow,nEpisodes,1);

% cancellation times are uniformly distributed in the interval [bookingTime, flightDeparture]
classCancellationsFunc = @(fclass) arrayfun(@(eps) binornd(1, fClassCancelRate(fclass), 1, nArrivals(eps,fclass)).*rand(1, nArrivals(eps,fclass)), 1:nEpisodes, 'UniformOutput', false)';
cancellations = arrayfun(@(fclass) classCancellationsFunc(fclass), 1:nWTPgroups, 'UniformOutput', false);
cancellations = horzcat(cancellations{:});

% nCancellations(flight episode Index, Fare Class Index)
nCancellations = cellfun(@(c) sum(ceil(c)), cancellations); % size = 1x1 cell -> 25000 x 3 array 

% Computing max reward for each episode by considering WTP of committed passengers
% maxReward(flightEpsIndex) = theoretical upper bound of revenue (maximum possible revenue) in that episode
% nArrivals - nCancellations = number of confirmed passengers
% nArrivals = cell2mat(nArrivalsCellArray); % 25000x6 vector
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

% episode 'j', fare class 'i' (bookingHorizon-arrivalTimes{j,i}).*cancellations{j,i} = time remaining*cancellation = cancellation time after (wrt) arrival time    
computeCancellationTimeFunc = @(epsInd, wtpGrp) ((bookingHorizon-arrivalTimes{epsInd, wtpGrp}).*cancellations{epsInd, wtpGrp} + arrivalTimes{epsInd, wtpGrp}.*ceil(cancellations{epsInd, wtpGrp}))'; % nArrivals(given episode and fare class)x1

compileTrDataFunc = @(epsInd) cell2mat(arrayfun(@(wtpGrp) [arrivalTimes{epsInd, wtpGrp}', wtpGrp*ones(nArrivals(epsInd, wtpGrp), 1), computeCancellationTimeFunc(epsInd, wtpGrp)], 1:nWTPgroups, 'UniformOutput', false)');
trData = arrayfun(@(epsInd) compileTrDataFunc(epsInd), 1:nEpisodes, 'UniformOutput', false); % 1x25000 cell array

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
