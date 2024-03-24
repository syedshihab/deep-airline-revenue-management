% This script is used for plotting the beta distributions modeling the fare class arrivals and getting the lambdaPrime and lambda values corresponding to each fare class and time step
% Author: Syed A.M. Shihab


% case: two fare classes
nFareClasses = 2;
bookingHorizon = 10; % time frames/DCPs/time steps = 182 days
fClassMeanDemand = [10, 30]; % [15, 60];
% capacity = 30
t = 0:1/100:1; % x-values in the beta pdf plot

% beta distribution shape parameters (alpha and beta) for each fare class
% fClass1 and fClass2 peak arrivals occur at t = 90% and 60% of bookingHorizon respectively; 
% so means of beta distributions = 0.90 and 0.60 respectively; mean = alpha/(alpha+beta)
alpha = [9*3, 6*3];  
beta = [10*3-alpha(1), 10*3-alpha(2)]; % beta = 10*3-alpha

% lambaPrimeValues = area under beta distribution curve at each time step
betaPDFvalues_fClasses = [betapdf(t,alpha(1),beta(1));... % betapdf(t,18,2);... fc1; 
                       betapdf(t,alpha(2),beta(2))];   % fc2
                       

for ind=1:nFareClasses
    subplot(nFareClasses,1,ind);      
    plot(t,betaPDFvalues_fClasses(ind,:))
    xlabel('time in booking period')
    ylabel('beta pdf values')    
end

% for each fare class, lambda(t) = lambdaPrime(t)*meanArrivalsPerEps
% the beta distribution is used to get time-varying lambdaPrime values at each time step
% lambdaPrimeValues = area under beta curve in each time step; total area = 1; 
lambdaPrimeValues = zeros(nFareClasses,bookingHorizon);
lambdaValues = zeros(nFareClasses,bookingHorizon);
for i=1:nFareClasses
    lambdaPrimeValues(i,:) = betacdf((1:bookingHorizon)/bookingHorizon,alpha(i),beta(i)) - betacdf((0:(bookingHorizon-1))/bookingHorizon,alpha(i),beta(i));
    lambdaValues(i,:) = lambdaPrimeValues(i,:)*fClassMeanDemand(i);
end

save('lambdaValues.mat','lambdaValues','-v7')

% totalAreaUnderBetaCurve = sum(lambdaPrimeValues(anyRow,:),2) = 1
% sum(lambdaValues(1,:),2) = fClassMeanDemand(1) = 15
% sum(lambdaValues(2,:),2) = fClassMeanDemand(2) = 60
