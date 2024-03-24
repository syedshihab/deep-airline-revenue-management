% 
% Author: Syed A.M. Shihab

% Type of problem: IQP 

% Standard form of optimization problem (as followed by Gurobi)
% obj: xQx + obj*x + alpha 	quadratic
% s.t. Ax = B 				linear constraints
%	   l <= x <= u			bound constraints
%	   all x integral		integrality constraints 


% In order to use Gurobi's MATLAB interface, you'll need to use the MATLAB function gurobi_setup to tell MATLAB where to find the Gurobi mex file.
% >> cd c:/gurobi810/win64/matlab
% >> gurobi_setup 
% The gurobi_setup function adjusts your MATLAB path to include the <installdir>/matlab directory. If you want to avoid typing this command every time you start MATLAB, 
% follow the instructions issued by gurobi_setup to permanently adjust your path.

% TODO: DO ALL THE TODOs
% TODO: DONE generate stochastic on-demand demand; d_arc ~ N(meanDemand,sigma); 
% then calculate average, variances and/or confidence intervals of profits generated during the operational/simulaiton period by the three different forms of service
% TODO: run simulation several times to calculate average daily weekday profit; each simulation will have different actualDemandVector because arc demands are sampled from a normal distribution
% TODO: vary sigma; if sigma is large, ondemand/hybrid services will generate more profit than scheduled services


clear all

% % adding the path of Gurobi directory
% addpath(genpath('C:\gurobi902\win64\matlab'))

%% List of parameters 

% fileName_paxArvData = 'testData_cnc' + string(1) + '_fcArrivals' + string(1) + '_2fp22pp_true_mktSiml'; 
% fileName_saveVars = 'maxRevenue_eps_trueMktSiml_testData.mat';
fileName_paxArvData = 'trData_cnc' + string(1) + '_fcArrivals' + string(1) + '_2fp22pp_mktSiml_mktEst';  
fileName_saveVars = 'maxRevenue_eps_mktSiml_mktEst_trData.mat';

load(fileName_paxArvData,'nArrivals','nCancellations_by_wtpGroup_time_eps','nEpisodes','trData','nWTPgroups','bookingHorizon');

if nEpisodes>30000
    nEpisodes=30000; % compute max revenue of at most 30000 episodes
end

% computing nCancellations_wtpGroup_arvTime_eps
nArrivals_wtpGroup_time_eps = nArrivals(:,:,1:nEpisodes);
nCancellations_wtpGroup_arvTime_eps = zeros(nWTPgroups,bookingHorizon,nEpisodes);
for eps=1:nEpisodes
    rowInd = find(trData{eps}(:,3));
    arvTimeInd = trData{eps}(rowInd,1);
    wtpGrpInd = trData{eps}(rowInd,2);
    for ind=1:length(rowInd)
        nCancellations_wtpGroup_arvTime_eps(wtpGrpInd(ind),arvTimeInd(ind),eps) =  nCancellations_wtpGroup_arvTime_eps(wtpGrpInd(ind),arvTimeInd(ind),eps)+1;
    end
end

nCommittedPassengers_wtpGroup_time_eps = nArrivals_wtpGroup_time_eps - nCancellations_wtpGroup_arvTime_eps;


nTimeSteps = bookingHorizon;
nPP_HFP = 2; % number of available price points for the high fare product
nPP_LFP = 2; % number of available price points for the low fare product
BCF = 2; % bumping cost factor
first=1;
last=2;

nDVs_pHpLpLnOBnBB = [nTimeSteps, nTimeSteps, nTimeSteps, 1, nPP_HFP, nPP_LFP]; 

p_H6_firstLastIndices = [1, nTimeSteps];                                                            % [1,nTimeSteps]; 
p_L2_firstLastIndices = [p_H6_firstLastIndices(last)+1, p_H6_firstLastIndices(last)+nTimeSteps];    % [nTimeSteps+1, 2*nTimeSteps];
p_L1_firstLastIndices = [p_L2_firstLastIndices(last)+1, p_L2_firstLastIndices(last)+nTimeSteps];    % [2*nTimeSteps+1, 3*nTimeSteps];
nOB_firstLastIndices = [p_L1_firstLastIndices(last)+1, p_L1_firstLastIndices(last)+1];              % [3*nTimeSteps+1, 3*nTimeSteps+1];
nBmp_H_firstLastIndices = [nOB_firstLastIndices(last)+1, nOB_firstLastIndices(last)+nPP_HFP];       % [3*nTimeSteps+2, 3*nTimeSteps+1+2];
nBmp_L_firstLastIndices = [nBmp_H_firstLastIndices(last)+1, nBmp_H_firstLastIndices(last)+nPP_LFP]; % [3*nTimeSteps+3+1, 3*nTimeSteps+3+2];


nDVs = nBmp_L_firstLastIndices(last); % total number of decision variables = sum(nDVs_pHpLpLnOBnBB)

calculateIndex_p_H6 = @(tStep) p_H6_firstLastIndices(first) - 1 + tStep;
calculateIndex_p_L2 = @(tStep) p_L2_firstLastIndices(first) - 1 + tStep;
calculateIndex_p_L1 = @(tStep) p_L1_firstLastIndices(first) - 1 + tStep;
calculateIndex_nOB = nOB_firstLastIndices(first);
calculateIndex_nBmp_H = @(pPoint) nBmp_H_firstLastIndices(first) - 1 + pPoint; % pPoint = {1,2}
calculateIndex_nBmp_L = @(pPoint) nBmp_L_firstLastIndices(first) - 1 + pPoint; % pPoint = {1,2}

% price points (pPoints)
H6_pp = 1;
H4_pp = 2;
L2_pp = 1;
L1_pp = 2;

% fare paid indices
H6_WTP = 1;
H4_WTP = 2;
L2_WTP = 3;
L1_WTP = 4;

maxRevenue_eps = zeros(nEpisodes,1);
resultLogger_eps = zeros(nEpisodes,nDVs);

for eps=1:nEpisodes

    %% constraints
    % Ax = B	linear constraints
    % Ax <= B	linear constraints
    % Ax >= B	linear constraints

    
    % Constraint 1: nOB >= nB - 100
    % nOB - nB >= -100    
    A1 = zeros(1,nDVs);
    
    A1(calculateIndex_nOB) = 1; % coefficient of nOB = 1;
    for tStepInd=1:nTimeSteps
        A1(calculateIndex_p_H6(tStepInd)) = nCommittedPassengers_wtpGroup_time_eps(H4_WTP,tStepInd,eps);
        A1(calculateIndex_p_L2(tStepInd)) = -nCommittedPassengers_wtpGroup_time_eps(L2_WTP,tStepInd,eps);
        A1(calculateIndex_p_L1(tStepInd)) = -nCommittedPassengers_wtpGroup_time_eps(L2_WTP,tStepInd,eps) - nCommittedPassengers_wtpGroup_time_eps(L1_WTP,tStepInd,eps);
    end

    A1_rhs = -100 + sum(nCommittedPassengers_wtpGroup_time_eps(H4_WTP,1:182,eps)) + sum(nCommittedPassengers_wtpGroup_time_eps(H6_WTP,1:182,eps)); % n_H6_1 + ... + n_H6_182 + n_H4_1 + ... n_H4_182
    A1_sense = '>';
    
    % Constraint 2: nbmp_L1 <= nOB
    % nbmp_L1 - nOB <= 0
    A2 = zeros(1,nDVs);
    A2(calculateIndex_nOB) = -1; % coefficient of nOB = -1;
    A2(calculateIndex_nBmp_L(L1_pp)) = 1; % coefficient of nBmp_L1 = 1;
    
    A2_rhs = 0; 
    A2_sense = '<';

    
    % Constraint 3: nBmp_L1 <= nB_L1
    % nBmp_L1 - nB_L1 <= 0
    A3 = zeros(1,nDVs);    
    A3(calculateIndex_nBmp_L(L1_pp)) = 1; % coefficient of nBmp_L1 = 1;
    for tStepInd=1:nTimeSteps        
        A3(calculateIndex_p_L1(tStepInd)) = -nCommittedPassengers_wtpGroup_time_eps(L1_WTP,tStepInd,eps);
    end
    
    A3_rhs = 0; 
    A3_sense = '<';
    
    
    % Constraint 4: nBmp_L2 <= nB_L2
    % nBmp_L2 - nB_L2 <= 0
    A4 = zeros(1,nDVs);    
    A4(calculateIndex_nBmp_L(L2_pp)) = 1; % coefficient of nBmp_L2 = 1;
    for tStepInd=1:nTimeSteps        
        A4(calculateIndex_p_L1(tStepInd)) = -nCommittedPassengers_wtpGroup_time_eps(L2_WTP,tStepInd,eps);
        A4(calculateIndex_p_L2(tStepInd)) = -nCommittedPassengers_wtpGroup_time_eps(L2_WTP,tStepInd,eps);
    end
    
    A4_rhs = 0; 
    A4_sense = '<';
    
    
    % Constraint 5: nBmp_H4 <= nB_H4
    % nBmp_H4 - nB_H4 <= 0
    A5 = zeros(1,nDVs);    
    A5(calculateIndex_nBmp_H(H4_pp)) = 1; % coefficient of nBmp_L1 = 1;
    for tStepInd=1:nTimeSteps        
        A5(calculateIndex_p_H6(tStepInd)) = nCommittedPassengers_wtpGroup_time_eps(H4_WTP,tStepInd,eps);
    end
    
    A5_rhs = sum(nCommittedPassengers_wtpGroup_time_eps(H4_WTP,1:182,eps)); 
    A5_sense = '<';
    
    
    % Constraint 6: nBmp_H6 <= nB_H6    
    A6 = zeros(1,nDVs);    
    A6(calculateIndex_nBmp_H(H6_pp)) = 1; % coefficient of nBmp_L1 = 1;
    A6_rhs = sum(nCommittedPassengers_wtpGroup_time_eps(H6_WTP,1:182,eps)); 
    A6_sense = '<';
    
    
    % Constraint 7: nBmp_L1 <= nB_L1
    A7 = zeros(1,nDVs);    
    A7(calculateIndex_nOB) = 1; % coefficient of nBmp_L1 = 1;
    A7(calculateIndex_nBmp_L(L1_pp)) = -1; % coefficient of nBmp_L1 = 1;
    A7(calculateIndex_nBmp_L(L2_pp)) = -1; % coefficient of nBmp_L1 = 1;
    A7(calculateIndex_nBmp_H(H4_pp)) = -1; % coefficient of nBmp_L1 = 1;
    A7(calculateIndex_nBmp_H(H6_pp)) = -1; % coefficient of nBmp_L1 = 1;
      
    A7_rhs = 0; 
    A7_sense = '=';
    
    
    % Constraint 8: p_L1_t + p_L2_t <= 1
    nConstraints = nTimeSteps; 
    A8 = zeros(nConstraints,nDVs); 
        
    for tStepInd=1:nTimeSteps % for constrInd=1:nTimeSteps        
        % constrInd = tStepInd
        A8(tStepInd,calculateIndex_p_L1(tStepInd)) = 1;
        A8(tStepInd,calculateIndex_p_L2(tStepInd)) = 1;
    end
    
    A8_rhs = ones(nConstraints,1); 
    A8_sense = repmat('<',nConstraints,1);
    
    
    %% Creating the optimization model 

    % Standard form of optimization problem (as followed by Gurobi)
    % obj: xQx + obj*x + alpha 	quadratic
    % s.t. Ax = B 				linear constraints
    %	   l <= x <= u			bound constraints
    %	   all x integral		integrality constraints 

    
    % Defining variable types    
    model.vtype = [repmat('B',1,3*nTimeSteps) repmat('I',1,1) repmat('I',1,nPP_HFP) repmat('I',1,nPP_LFP)];
    % price variables p_ij - continuous, w_ij variables - integers, x_ij_k and x_i_k variables are binary, and e_j_k, g_ij_k, r_ij_uk and r_ij_dk variables are continuous


    %% Specifying variable upper bounds
    % model.ub = [repmat(1e6,1,nArcs) repmat(capeVTOLs*neVTOLs,1,nArcs) repmat(1,1,neVTOLs*nArcs) repmat(1,1,neVTOLs*nNodes)]; 
    model.ub = [repmat(1,1,3*nTimeSteps),... % p_ij_k_ub = 80
                repmat(100,1,1),... % w_ij_ub = 4*neVTOLs; w_ij will take this value in the extreme case where all the vehicles are assigned to arc ij and demand>=4*neVTOLs
                repmat(100,1,nPP_HFP),... % x_ij_k_ub = 1
                repmat(100,1,nPP_LFP)]; % x_i_k_ub = 1
    

    %% Specifying variable lower bounds
    model.lb = zeros(1,nDVs);
    

    %% Defining the objective function

    % linear terms
    model.obj = zeros(1,nDVs); % Populating the obj vector - the linear objective vector - of the objective function
    for tStepInd=1:nTimeSteps        
        model.obj(calculateIndex_p_H6(tStepInd)) = 2*nCommittedPassengers_wtpGroup_time_eps(H6_WTP,tStepInd,eps)-4*nCommittedPassengers_wtpGroup_time_eps(H4_WTP,tStepInd,eps);
        model.obj(calculateIndex_p_L2(tStepInd)) = 2*nCommittedPassengers_wtpGroup_time_eps(L2_WTP,tStepInd,eps);
        model.obj(calculateIndex_p_L1(tStepInd)) = 1.5*nCommittedPassengers_wtpGroup_time_eps(L2_WTP,tStepInd,eps) + 1.5*nCommittedPassengers_wtpGroup_time_eps(L1_WTP,tStepInd,eps);
    end
    model.obj(calculateIndex_nBmp_L(L1_pp)) = -1.5*BCF;
    model.obj(calculateIndex_nBmp_L(L2_pp)) = -2*BCF;
    model.obj(calculateIndex_nBmp_H(H4_pp)) = -4*BCF;
    model.obj(calculateIndex_nBmp_H(H6_pp)) = -6*BCF;
   
    model.objcon = 4*sum(nCommittedPassengers_wtpGroup_time_eps(H4_WTP,1:182,eps)) + 4*sum(nCommittedPassengers_wtpGroup_time_eps(H6_WTP,1:182,eps));
    
    model.modelsense = 'max'; 
    
    model.A = sparse([A1; A2; A3; A4; A5; A6; A7; A8]);
    model.rhs = [A1_rhs; A2_rhs; A3_rhs; A4_rhs; A5_rhs; A6_rhs; A7_rhs; A8_rhs];
    model.sense = [A1_sense; A2_sense; A3_sense; A4_sense; A5_sense; A6_sense; A7_sense; A8_sense];
    

    % params.PreQLinearize = 2
    % params.Presolve = 1;
    % results = gurobi(model,params)
    % results = gurobi_relax(model)
    % result = gurobi(model);

    % params.NonConvex = 2;
    result = gurobi(model);
    maxRevenue_eps(eps) = result.objval;
    resultLogger_eps(eps,:) = result.x;

    % for v=1:length(names)
    %     fprintf('%s %e\n', names{v}, results.x(v));
    % end
    % 
    % fprintf('Profit: %e\n', results.objval);
end

save(fileName_saveVars,'maxRevenue_eps','resultLogger_eps');

% % Verifying results
% load(fileName_paxArvData,'maxReward');
% diff_reward_eps = maxReward - maxRevenue_eps;
% max_diff_reward = max(diff_reward_eps);
% ind_episodesWithOB = find(resultLogger_eps(:,547)); % calculateIndex_nOB 