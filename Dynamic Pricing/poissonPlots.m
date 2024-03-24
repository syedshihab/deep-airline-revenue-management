x = 0:1:100;
fClassDistrMeans = [8, 12, 15, 15, 40, 60];

nArrivals_wtpGrps = [poisspdf(x,fClassDistrMeans(1));...
                     poisspdf(x,fClassDistrMeans(2));...
                     poisspdf(x,fClassDistrMeans(3));...
                     poisspdf(x,fClassDistrMeans(4));...
                     poisspdf(x,fClassDistrMeans(5));...
                     poisspdf(x,fClassDistrMeans(6))];

for ind=1:6
    subplot(2,3,ind);      
    plot(x,nArrivals_wtpGrps(ind,:))
end
