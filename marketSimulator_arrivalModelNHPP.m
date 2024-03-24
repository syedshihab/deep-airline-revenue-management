clear all;
close all;

% TODO: make totalDemand = 1.5*capacity = 150 if capacity=100
% TODO: new arrival model based on NHPP and perhaps new cancellation model also based on MHPP
% TODO: change number of actions;

% Parameters % TODO: use real world problem parameters

capacity = 100; % total demand = capacity*1.5 = 300 % 100, total demand = 150; % flight capacity (of economy cabin); vary
bookingHorizon = 182; % 182 days = 6 months % 365 % 1000; % total booking horizon/window/interval
% nFclassPricePoints = [2 2 2]; [3 3 3];

% TODO: implement the polya process to model the arrival of passenger booking requests

nFareClasses = 3; % high, medium and low fare class; each class has two or more distinct price points associated with it
% fClassBookingWindow = [repelem(14,fClassnPricePoints(1)) repelem(28,fClassnPricePoints(2)) repelem(182,fClassnPricePoints(3))]; % TODO: vary
% wtpGroupArrivalWindow = [7 14 21 28 100 182];

nEpisodes = 100000 ; % 50000; % 25000; % number of episodes used for training the DRL agent
% fClassPrices = {[600, 500, 400], [300, 250, 200], [150, 125, 100]}; % price will be set by agent
fClassPrices = {[500, 400], [250, 200], [130, 100]}; % price will be set by agent
nFclassPricePoints = cellfun(@length,fClassPrices);
nWTPgroups = sum(nFclassPricePoints); % nWTPgroups = 9; % 6; % high, medium and low fare class; each class may have a set of price points associated with it
fclassIdentifier = [repelem(1,nFclassPricePoints(1)) repelem(2,nFclassPricePoints(2)) repelem(3,nFclassPricePoints(3))]; % TODO: vary; # [1 1 2 2 3 3];
fclassIdentifierPy = fclassIdentifier - 1; % [0 0 1 1 2 2]
wtp = [fClassPrices{1} fClassPrices{2} fClassPrices{3}]; 
fareLevels = wtp; % = [400, 300, 250, 200, 130, 100]


% priceMatrix = zeros(prod(nFclassPricePoints+1),nUnqFareClasses);
priceMatrix = zeros(prod([nFclassPricePoints(1) nFclassPricePoints(2:end)+1]),nFareClasses);
rowInd = 1;
for p1=flip(fClassPrices{1}) % [300 400] % most expensive fare class will never be closed; TODO: remove inf; before doing so, test if agent learns this rule
    for p2=[flip(fClassPrices{2}) inf] % 0:2
        for p3=[flip(fClassPrices{3}) inf] % 0:2 
            priceMatrix(rowInd,:)=[p1 p2 p3];
            rowInd = rowInd + 1;
        end
    end
end

% fClassDistrMeans = [10, 30, 60; 60, 30, 10; 33, 33, 34]; % three different settings of fare class passenger distributions 
fClassDistrMeans = [8, 12, 15, 15, 30, 70]; % mean number of arrivals of passengers in different WTP groups/categories; % totalDemand = 1.5*capacity = 150 if capacity=100 
% -> mean number of passenger arrivals of each fare class in each flight episode in three different scenarios/cases
% In row 1, mean(high class pax arrival) = 10, mean(middle class pax arrival) = 30 and mean(low class pax arrival) = 60 
       
fd = 1; % fd = {1,2,3} fd gives the fare class passenger distribution setting 
fClassMeanDemand = fClassDistrMeans(fd,:); % here fclass = WTP mean demand
 
fClassCancelRate = [0.1, 0.1, 0.05, 0.05, 0, 0]; % booking/ticket cancellation rates are different for each fare class;
cncFee = [0 100]; % fClassCancelRefund = [1 0.5 0];
% fClassCancelRate = [0.1, 0.05, 0]; %TODO: booking/ticket cancellation rates should be different for each fare class; 
% lowest fare class passengers do not have the option of cancelling their booking

% Beta distributions represent the nonhomogeneous (varying) arrival rates of the different wtpGroups
t = 0:1/182:1;
betaValues_wtpGrps = [betapdf(t,9,4);...
                      betapdf(t,7,4);...
                      betapdf(t,6,4);...
                      betapdf(t,5,5);...
                      betapdf(t,4,6);...
                      betapdf(t,2,6)];

for ind=1:6
    subplot(2,3,ind);      
    plot(t,betaValues_wtpGrps(ind,:))
end


t = 0:1:300;
gammaValues_wtpGrps = [gampdf(t,150,1);... % variance = 150
                      gampdf(t,75,2);... % variance = 300
                      gampdf(t,50,3);... % variance = 450
                      gampdf(t,25,6);... % variance = 36*25 = 900
                      gampdf(t,200,3/4);... % variance = 200*9/16 = 112.5
                      gampdf(t,250,3/5)]; % variance = 250*9/25 = 90

for ind=1:6
    subplot(2,3,ind);      
    plot(t,gammaValues_wtpGrps(ind,:))
end

pcArrivals_wtpGrps = fClassDistrMeans/150;

fileName = 'trainingData_c1fd1_NHPP1'; 

lamdaValues_wtpGrps = betaValues_wtpGrps.*(A*pcArrivals_wtpGrps');

% Assuming that passengers arrive following a Poisson process with mean number of arrivals given by fClassMeanDemand
avgInterArrivalTime = wtpGroupArrivalWindow./fClassMeanDemand; % [14 21 182]./[10 30 60] size = 1 x 3

% %%%% TODO
% % Simulating arrival of low fare class passengers
% maxNumberOfArrivals = poissinv(1-1/(10*nEpisodes), fClassMeanDemand(end)); % -> poissinv(1, classSizeMean); size = 1 x 3
% arrivalTimesLclass = num2cell(cumsum(exprnd(avgInterArrivalTime(end), nEpisodes, maxNumberOfArrivals), 2), 2); % 25000x1 cell array; exprnd() size = 25000 x maxNumArrivals 
% % set arrival times beyond totalTime equal to 0
% arrivalTimesLclass = cellfun(@(c) c(c < bookingHorizon), arrivalTimesLclass, 'UniformOutput', false);
% % arrivalTimes = 1 x 1 cell -> 25000 x 3 cell array -> numeric array of arrival times 
% priceSensitivity = [-0.2 -0.5 -0.8];
% nArrivalsLclass = cellfun(@length, arrivalTimesLclass);
% nArrivalsLclass1 = max(0,nArrivalsLclass + priceSensitivity(end)*50); 
% nArrivalsLclass2 = max(0,nArrivalsLclass + priceSensitivity(end)*25) - nArrivalsLclass1; 
% nArrivalsLclass3 = nArrivalsLclass - nArrivalsLclass1 - nArrivalsLclass2;
% 
% y = arrayfun(@(i)randsample(nArrivalsLclass(i),nArrivalsLclass1(i)),1:nEpisodes,'UniformOutput', false); % indices of pax with fareMclass>WTP>fareLclass1
% %%%%

% TODO: This could get large as totalTime increases
% TODO: May need to increase max number of passengerArrivals to insure that totalTime is reached (i.e. passengers keep arriving till flight departure)
% nArrivalsCellArray = arrayfun(@(lambda)poissrnd(fClassMeanDemand(lambda),nEpisodes,1),1:nFareClasses,'UniformOutput', false); % size = 1 x 3 cell array
nArrivalsCellArray = arrayfun(@(lambda)poissrnd(fClassMeanDemand(lambda),nEpisodes,1),1:(nWTPgroups-1),'UniformOutput', false); % size = 1 x 3 cell array

% episode = 1
% get cell array of arrival times for a single fare class
% using a cell array is necessary to allow for possibly different number of arrivals at each flight/episode
% arrivalTimesFunc = @(fareClass) num2cell(cumsum(exprnd(avgInterArrivalTime(fareClass), 1, nArrivalsCellArray{fareClass}(episode)), 2), 2); % -> exprnd() size = 25000 x maxNumArrivals 

arrivalTimesFunc12 = @(episode) arrayfun(@(fareClass) (sort(bookingHorizon - (cumsum(exprnd(avgInterArrivalTime(fareClass), 1, nArrivalsCellArray{fareClass}(episode)), 2)), 2)),1:(nWTPgroups-1), 'UniformOutput', false); % -> exprnd() size = 25000 x maxNumArrivals 
% arrivalTimesFunc = @(episode) arrayfun(@(fareClass) num2cell(cumsum(exprnd(avgInterArrivalTime(fareClass), 1, nArrivalsCellArray{fareClass}(episode)), 2), 2),1:nFareClasses, 'UniformOutput', false); % -> exprnd() size = 25000 x maxNumArrivals 

% arrivalTimesFunc is a function handle of an anonymous function; 'i' is the input argument; num2cell gives a 25000 x 1 cell array in 'arrivalTimesFunc' 
% for each input fare class 'i'; arrivalTimesFunc(r) = cell(r) = numeric array of arrivalTimes of passengers for flight 'r'
% returns 25000x1 cell array

% flight episode index = episode number or flight number 
% fareClassIndex = {1,2,3}
% arrivalTimes{flight episode Index, Fare Class Index}(arrival Index), arrival Index < nArrivals
% arrivalTimes = arrayfun(@(fareClass) arrivalTimesFunc(fareClass) , 1:nFareClasses, 'UniformOutput', false);

arrivalTimes12 = arrayfun(@(episode) arrivalTimesFunc12(episode),(1:nEpisodes)', 'UniformOutput', false);

% Computing arrivalTimesLclass
maxNumberOfArrivals = poissinv(1-1/(10*nEpisodes), fClassMeanDemand(end)); % -> poissinv(1, classSizeMean); size = 1 x 3
arrivalTimesLclass = num2cell(cumsum(exprnd(avgInterArrivalTime(end), nEpisodes, maxNumberOfArrivals), 2), 2); % 25000x1 cell array; exprnd() size = 25000 x maxNumArrivals 
arrivalTimesLclass = cellfun(@(c) c(c < bookingHorizon), arrivalTimesLclass, 'UniformOutput', false);
nArrivalsLclass = cellfun(@length, arrivalTimesLclass); % 25000x1 double

nArrivalsCellArray = [nArrivalsCellArray nArrivalsLclass];
% arrivalTimesFunc2 = @(episode) cumsum(exprnd(avgInterArrivalTime(3), 1, nArrivalsCellArray{3}(episode)), 2); % -> exprnd() size = 25000 x maxNumArrivals 
% arrivalTimes2 = arrayfun(@(episode) arrivalTimesFunc2(episode),(1:nEpisodes)', 'UniformOutput', false);

% 'UniformOutput' = false, so that arrayfun returns the outputs of arrivalTimesFunc in cell arrays. The outputs of arrivalTimesFunc are of size 25000 x 1
% arrivalTimes is a 1x3 cell array 
% arrivalTimes(1) = cell(1) = arrivalTimesFunc(1) = 25000 x 1 cell array; 
% arrivalTimesFunc(r) = cell(r) = numeric array of arrivalTimes of passengers for flight 'r'
% horizontally concatenate the cells of arrivalTimes

arrivalTimes12 = vertcat(arrivalTimes12{:}); % arrivalTimes = 25000x3 cell array; arrivalTimes{2, 3}(5) = arrival time of fifth passenger for third fare class in flight/episode = 2  
% arrivalTimes = horzcat(arrivalTimes{:}); % arrivalTimes = 25000x3 cell array; arrivalTimes{2, 3}(5) = arrival time of fifth passenger for third fare class in flight/episode = 2 
arrivalTimes = [arrivalTimes12 arrivalTimesLclass];
 
% % set arrival times beyond totalTime (365 days) equal to 0
% arrivalTimes = cellfun(@(c) c(c < totalTime), arrivalTimes, 'UniformOutput', false);
% % arrivalTimes = 1 x 1 cell -> 25000 x 3 cell array -> numeric array of arrival times 
% 
% % nArrivals(flight episode index, Fare Class Index)
% nArrivals = cellfun(@length, arrivalTimes); 
% % nArrivals = 1 x 1 cell -> 25000 x 3 cell array -> length(numeric array of arrival times )
% % nArrivals{1}(i,j) = number of passenger arrivals of fare class 'j' in flight episode 'i', where i = flight episode index E {1,...,25000}; j = fare class index E {1,2,3}; 

% 
% cancellations{flight episode index, Fare Class Index}(arrival Index)
% zero for not canceled otherwise uniform between zero and one for percentage
% of time left before cancellation
% binornd(1, fClassCancelRate(j), 1, nArrivals(i, j)) = numeric array of random 0's and 1's sampled from the binomial distribution 
% rand(1, nArrivals(i, j)) = random number array of size 1 x nArrivals
% binornd(1, fClassCancelRate(j), 1, nArrivals(i, j)).*rand(1, nArrivals(i, j)) = array of 0's and uniformly random cancellation times in the interval [0,1]

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

% % TODO: compute revenues generated from using EMSRb method
% bLimits = py.EMSRbBLimits(fareLevels,fClassMeanDemand,capacity);



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
    trDataPy{eps}((trDataPy{eps}(:,1)<0),1)=0; % prcArr(prcArr>10)=10
end
  
% save all variables in the workspace
save(strcat(fileName, '.mat')) % -> filename.mat
% load(strcat(fileName, '.mat'))
