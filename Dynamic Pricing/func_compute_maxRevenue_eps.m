function [maxRevenue_eps,result,model] = func_compute_maxRevenue_eps(nArrivals_wtpGroup_time_eps, nCancellations_wtpGroup_time_eps, nEpisodes)

% nArrivals_wtpGroup_time_eps = nArrivals;
% nCancellations_wtpGroup_time_eps = nCancellations_by_wtpGroup_time_eps;
nCommittedPassengers_wtpGroup_time_eps = nArrivals_wtpGroup_time_eps - nCancellations_wtpGroup_time_eps;
nTimeSteps = 182;
nPP_HFP = 2;
nPP_LFP = 2;
BCF = 2;
first=1;
last=2;

nDVs_pHpLpLnOBnBB = [nTimeSteps, nTimeSteps, nTimeSteps, 1, 4]; 

p_H6_firstLastIndices = [1, nTimeSteps];                                                            % [1,nTimeSteps]; 
p_L2_firstLastIndices = [p_H6_firstLastIndices(last)+1, p_H6_firstLastIndices(last)+nTimeSteps];    % [nTimeSteps+1, 2*nTimeSteps];
p_L1_firstLastIndices = [p_L2_firstLastIndices(last)+1, p_L2_firstLastIndices(last)+nTimeSteps];    % [2*nTimeSteps+1, 3*nTimeSteps];
nOB_firstLastIndices = [p_L1_firstLastIndices(last)+1, p_L1_firstLastIndices(last)+1];              % [3*nTimeSteps+1, 3*nTimeSteps+1];
nBmp_H_firstLastIndices = [nOB_firstLastIndices(last)+1, nOB_firstLastIndices(last)+nPP_HFP];       % [3*nTimeSteps+2, 3*nTimeSteps+1+2];
nBmp_L_firstLastIndices = [nBmp_H_firstLastIndices(last)+1, nBmp_H_firstLastIndices(last)+nPP_LFP]; % [3*nTimeSteps+3+1, 3*nTimeSteps+3+2];


nDVs = nBmp_L_firstLastIndices(last); % total number of decision variables

calculateIndex_p_H6 = @(tStep) p_H6_firstLastIndices(first) - 1 + tStep;
calculateIndex_p_L2 = @(tStep) p_L2_firstLastIndices(first) - 1 + tStep;
calculateIndex_p_L1 = @(tStep) p_L1_firstLastIndices(first) - 1 + tStep;
calculateIndex_nOB = nOB_firstLastIndices(first);
calculateIndex_nBmp_H = @(pPoint) nBmp_H_firstLastIndices(first) - 1 + pPoint; % pPoint = {1,2}
calculateIndex_nBmp_L = @(pPoint) nBmp_L_firstLastIndices(first) - 1 + pPoint; % pPoint = {1,2}

H6 = 1;
H4 = 2;
L2 = 1;
L1 = 2;

H6_WTP = 1;
H4_WTP = 2;
L2_WTP = 3;
L1_WTP = 4;

for eps=1:nEpisodes

    %% constraints
    % Ax = B	linear constraints
    % Ax <= B	linear constraints
    % Ax >= B	linear constraints

    % d_ij = q_ij - beta*p_ij % price responsive demand

    % Inequality constraint
    % Capacity constraint on w_ij (number of constraints = nArcs): w_ij <= combined capacity of eVTOLs in arc ij
    % w_ij - 4*sum_over_k(x_ij_k) <= 0 for all arcs

    % Constraint 1
    % nOB - nB >= -100
    A1 = zeros(1,nDVs);
    A1(calculateIndex_nOB) = 1; % coefficient of nOB = 1;
    for tStepInd=1:nTimeSteps
        A1(calculateIndex_p_H6(tStepInd)) = nCommittedPassengers_wtpGroup_time_eps(H4_WTP,tStepInd,eps);
        A1(calculateIndex_p_L2(tStepInd)) = -nCommittedPassengers_wtpGroup_time_eps(L2_WTP,tStepInd,eps);
        A1(calculateIndex_p_L1(tStepInd)) = -nCommittedPassengers_wtpGroup_time_eps(L2_WTP,tStepInd,eps) - nCommittedPassengers_wtpGroup_time_eps(L1_WTP,tStepInd,eps);
    end

    A1_rhs = -100 + sum(nCommittedPassengers_wtpGroup_time_eps(H4_WTP,1:182,eps)) + sum(nCommittedPassengers_wtpGroup_time_eps(H6_WTP,1:182,eps)); % n_H6_1 + ... + n_H6_182 + n_H4_1 + ... n_H4_182
    A1_sense = repmat('>',nArcs,1);
    
    % Constraint 2
    % nbmp_L1 <= nOB
    A2 = zeros(1,nDVs);
    A2(calculateIndex_nOB) = -1; % coefficient of nOB = -1;
    A2(calculateIndex_nBmp_L(L1)) = 1; % coefficient of nBmp_L1 = 1;
    
    A2_rhs = 0; 
    A2_sense = repmat('<',nArcs,1);

    
    % Constraint 3
    % nBmp_L1 <= nB_L1
    A3 = zeros(1,nDVs);    
    A3(calculateIndex_nBmp_L(L1)) = 1; % coefficient of nBmp_L1 = 1;
    for tStepInd=1:nTimeSteps        
        A3(calculateIndex_p_L1(tStepInd)) = -nCommittedPassengers_wtpGroup_time_eps(L1_WTP,tStepInd,eps);
    end
    
    A3_rhs = 0; 
    A3_sense = repmat('<',nArcs,1);
    
    
    % Constraint 4
    % nBmp_L2 <= nB_L2
    A4 = zeros(1,nDVs);    
    A4(calculateIndex_nBmp_L(L2)) = 1; % coefficient of nBmp_L2 = 1;
    for tStepInd=1:nTimeSteps        
        A4(calculateIndex_p_L1(tStepInd)) = -nCommittedPassengers_wtpGroup_time_eps(L2_WTP,tStepInd,eps);
        A4(calculateIndex_p_L2(tStepInd)) = -nCommittedPassengers_wtpGroup_time_eps(L2_WTP,tStepInd,eps);
    end
    
    A4_rhs = 0; 
    A4_sense = repmat('<',nArcs,1);
    
    
    % Constraint 5
    % nBmp_H4 <= nB_H4
    A5 = zeros(1,nDVs);    
    A5(calculateIndex_nBmp_H(H4)) = 1; % coefficient of nBmp_L1 = 1;
    for tStepInd=1:nTimeSteps        
        A5(calculateIndex_p_H6(tStepInd)) = nCommittedPassengers_wtpGroup_time_eps(H4_WTP,tStepInd,eps);
    end
    
    A5_rhs = sum(nCommittedPassengers_wtpGroup_time_eps(H4_WTP,1:182,eps)); 
    A5_sense = repmat('<',nArcs,1);
    
    
    % Constraint 6
    % nBmp_L1 <= nB_L1
    A6 = zeros(1,nDVs);    
    A6(calculateIndex_nBmp_H(H6)) = 1; % coefficient of nBmp_L1 = 1;
      
    A6_rhs = sum(nCommittedPassengers_wtpGroup_time_eps(H6_WTP,1:182,eps)); 
    A6_sense = repmat('<',nArcs,1);
    
    
    % Constraint 7
    % nBmp_L1 <= nB_L1
    A7 = zeros(1,nDVs);    
    A7(calculateIndex_nOB) = 1; % coefficient of nBmp_L1 = 1;
    A7(calculateIndex_nBmp_L(L1)) = -1; % coefficient of nBmp_L1 = 1;
    A7(calculateIndex_nBmp_L(L2)) = -1; % coefficient of nBmp_L1 = 1;
    A7(calculateIndex_nBmp_H(H4)) = -1; % coefficient of nBmp_L1 = 1;
    A7(calculateIndex_nBmp_H(H6)) = -1; % coefficient of nBmp_L1 = 1;
      
    A7_rhs = 0; 
    A7_sense = repmat('=',nArcs,1);
    
    




    %% Creating the optimization model 

    % Standard form of optimization problem (as followed by Gurobi)
    % obj: xQx + obj*x + alpha 	quadratic
    % s.t. Ax = B 				linear constraints
    %	   l <= x <= u			bound constraints
    %	   all x integral		integrality constraints 

    % model.vtype = [repmat('C',1,nArcs) repmat('I',1,nArcs) repmat('B',1,neVTOLs*nArcs) repmat('B',1,neVTOLs*nNodes)]; % price variables - continuous, w variables - integers, x variables are binary
    % price variables - continuous, w variables - integers, x variables are binary

    % Defining variable types
    % nDVs_pwxxegrurd = [nArcs, nArcs, neVTOLs*nArcs, neVTOLs*nNodes, neVTOLs*nSTnodes, neVTOLs*nGarcs, neVTOLs*nGarcs, neVTOLs*nGarcs]; 
    model.vtype = [repmat('B',1,3*nTimeSteps) repmat('I',1,1) repmat('I',1,2) repmat('I',1,2)];
    % price variables p_ij - continuous, w_ij variables - integers, x_ij_k and x_i_k variables are binary, and e_j_k, g_ij_k, r_ij_uk and r_ij_dk variables are continuous


    %% Specifying variable upper bounds
    % model.ub = [repmat(1e6,1,nArcs) repmat(capeVTOLs*neVTOLs,1,nArcs) repmat(1,1,neVTOLs*nArcs) repmat(1,1,neVTOLs*nNodes)]; 
    model.ub = [repmat(1,1,3*nTimeSteps),... % p_ij_k_ub = 80
                repmat(100,1,1),... % w_ij_ub = 4*neVTOLs; w_ij will take this value in the extreme case where all the vehicles are assigned to arc ij and demand>=4*neVTOLs
                repmat(100,1,2),... % x_ij_k_ub = 1
                repmat(100,1,2)]; % x_i_k_ub = 1
    

    %% Specifying variable lower bounds
    model.lb = zeros(1,nDVs);
    

    %% Defining the objective function

    % linear terms
    model.obj = zeros(1,nDVs); % model.c % Populating the obj vector - the linear objective vector - of the objective function
    for tStepInd=1:nTimeSteps        
        model.obj(calculateIndex_p_H6(tStepInd)) = 2*nCommittedPassengers_wtpGroup_time_eps(H6_WTP,tStepInd,eps)-4*nCommittedPassengers_wtpGroup_time_eps(H4_WTP,tStepInd,eps);
        model.obj(calculateIndex_p_L2(tStepInd)) = 2*nCommittedPassengers_wtpGroup_time_eps(L2_WTP,tStepInd,eps);
        model.obj(calculateIndex_p_L1(tStepInd)) = 1.5*nCommittedPassengers_wtpGroup_time_eps(L2_WTP,tStepInd,eps) + 1.5*nCommittedPassengers_wtpGroup_time_eps(L1_WTP,tStepInd,eps);
    end
    model.obj(calculateIndex_nBmp_L(L1)) = -1.5*BCF;
    model.obj(calculateIndex_nBmp_L(L2)) = -2*BCF;
    model.obj(calculateIndex_nBmp_H(H4)) = -4*BCF;
    model.obj(calculateIndex_nBmp_H(H6)) = -6*BCF;
   
    model.alpha = 4*sum(nCommittedPassengers_wtpGroup_time_eps(H4_WTP,1:182,eps)) + 4*sum(nCommittedPassengers_wtpGroup_time_eps(H6_WTP,1:182,eps));
    
    model.modelsense = 'max'; 


    % if dispatchInd==0
    model.A = sparse([A1; A2; A3; A4; A5; A6; A7]);
    model.rhs = [A1_rhs; A2_rhs; A3_rhs; A4_rhs; A5_rhs; A6_rhs; A7_rhs];
    model.sense = [A1_sense; A2_sense; A3_sense; A4_sense; A5_sense; A6_sense; A7_sense];
    

    % params.PreQLinearize = 2
    % params.Presolve = 1;
    % results = gurobi(model,params)
    % results = gurobi_relax(model)
    % result = gurobi(model);

    % params.NonConvex = 2;
    result = gurobi(model,params);
    maxRevenue_eps = result.objVal;

    % for v=1:length(names)
    %     fprintf('%s %e\n', names{v}, results.x(v));
    % end
    % 
    % fprintf('Profit: %e\n', results.objval);
end