clear all

t = 0:1/182:1;


% beta distribution shape parameters for each fare class
% fClass1, fClass2 and fClass3 peak arrivals occur at t=0.95, 0.77 and 0.39 respectively; 
% so means of beta distributions = 0.95, 0.77 and 0.39 respectively
% alpha = [9.45*3, 8.5*3, 7.7*2, 6.5*2, 5.00*3, 3.85*3];  
% beta = [10*3-alpha(1), 10*3-alpha(2), 10*2-alpha(3), 10*2-alpha(4), 10*3-alpha(5), 10*3-alpha(6)];

alpha = [9.45*3, 8.5*3, 7.0*2, 6.5*2, 5.00*3, 2.85*3];  
beta = [10*3-alpha(1), 10*3-alpha(2), 10*2-alpha(3), 10*2-alpha(4), 10*3-alpha(5), 10*3-alpha(6)];

% lambaPrimeValues = betaValues of each fare class at each time step
% betaValues_fClasses = [betapdf(t,9.5*3,(10*3-9.5*3));... % betapdf(t,18,2);... fc1; 
%                        betapdf(t,7.7*2,(10*2-7.7*2));... % fc2
%                        betapdf(t,3.9*3,(10*3-3.9*3))]; % fc3

% for ind=1:3
%     subplot(3,1,ind);      
%     plot(t,betaValues_fClasses(ind,:))
% end

% nFareClasses = 3;
% fClassMeanDemand = [33, 33, 34];
% nEpisodes = 5;
% bookingHorizon = 10;


betaPDFvalues_fClasses = [betapdf(t,alpha(1),beta(1));... % betapdf(t,18,2);... fc1; 
                          betapdf(t,alpha(2),beta(2));...
                          betapdf(t,alpha(3),beta(3));...
                          betapdf(t,alpha(4),beta(4));...
                          betapdf(t,alpha(5),beta(5));...
                          betapdf(t,alpha(6),beta(6))];   % fc3

wtpGroupMeanArrivals = [8, 8, 20, 20, 40, 60];
% the beta distribution is used to get time-varying lambdaPrime values at each time step
% lambdaPrimeValues = area under beta curve in each time step; total area = 1; 
lambdaPrimeValues = zeros(6,182);
lambdaValues = zeros(6,182); % for each wtpGroup, lambda(t) = lambdaPrime(t)*meanNarrivals
for i=1:6
    lambdaPrimeValues(i,:) = betacdf((1:182)/182,alpha(i),beta(i)) - betacdf((0:181)/182,alpha(i),beta(i)); % area under beta curve at different time steps
    lambdaValues(i,:) = lambdaPrimeValues(i,:)*wtpGroupMeanArrivals(i); % average number of arrivals (Poisson means) at different time steps
end

lambdaValues = [lambdaValues zeros(6,1)]; % makes the plot look better
WTPgroupLabels = ["WTPgroup-1", "WTPgroup-2", "WTPgroup-3", "WTPgroup-4", "WTPgroup-5", "WTPgroup-6"];

% for fcInd=1:3
%     plot(t,betaPDFvalues_fClasses(fcInd,:),'DisplayName',strcat(fareClassLabels(1,fcInd),' fare class'))
%     xlabel('time in booking period')
%     ylabel('PDF')
%     hold on
% end
% legend('Orientation','horizontal')

t = 0:182;
for wtpInd=1:6
    plot(t,lambdaValues(wtpInd,:),'DisplayName',WTPgroupLabels(1,wtpInd))
    xlim([0 182])
    xlabel('Time in booking period (day)')
    ylabel('Arrival rate')
    hold on
end
legend('Orientation','horizontal')
hold off
saveas(gcf,'plot_fclassPaxArrival.pdf')

% % the beta distribution is used to get time-varying lambdaPrime values at each time step
% % lambdaPrimeValues = area under beta curve in each time step; total area = 1; 
% lambdaPrimeValues = zeros(3,182);
% lambdaValues = zeros(3,182);
% for i=1:nFareClasses
%     lambdaPrimeValues(i,:) = betacdf((1:182)/182,alpha(i),beta(i)) - betacdf((0:181)/182,alpha(i),beta(i));
%     lambdaValues(i,:) = lambdaPrimeValues(i,:)*fClassMeanDemand(i);
% end

