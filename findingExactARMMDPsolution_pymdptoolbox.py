import mdptoolbox
import numpy as np
from scipy.sparse import csr_matrix

nActions = 3 # 2 # 3
nStates = 182*(1821**3)*(11**2) # 10 # 182*100*100*100*10*10

##actionMatrix = [1 0 0;... % action=0
##                1 1 0;... % action=1
##                1 1 1];   % action=2

# p = np.zeros((nActions,nStates,nStates))

# if action = 0 = [1,0,0]
p1 = csr_matrix((nStates,nStates))
for rowInd=range(nStates):
    t,rem = divmod(rowInd,(1821**3*11**2))
    b1,rem = divmod(rem,(1821**2*11**2))
    b2,rem = divmod(rem,(1821*11**2))
    b3,rem = divmod(rem,(11**2))
    c1,c2 = divmod(rem,(11))
    
    if (t<182):
        colInd = (t+1)*(1821**3*11**2)
        p1[rowInd,colInd]=1
    else: # t=182 (terminal state)
        colInd = 0
        p1[rowInd,colInd]=1 
        

# if action = 1 = [1,1,0]
p2 = 

# if action = 2 = [1,1,1]
p3 =






##vi = mdptoolbox.mdp.ValueIteration(P, R, 1)
##vi.run()
##vi.policy # result is (0, 0, 0)
##
##fh = mdptoolbox.mdp.FiniteHorizon(P, R, 0.9, 3)
##fh.run()

##nStates = 0
##for x in range(11,1821):
##    nStates = nStates + x**3
##    
##print('nStates =',nStates)
