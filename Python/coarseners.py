import numpy as np
import scipy.sparse as sp

"""
MGM Prologation/Restricting Strategies
"""
def oned_prolong(nc):
    """Interpolator from ceil(n/2)-1 => n-1
    @param nc: number of course grid points
    """
    n = 2*(nc+1)
    nf = n-1
    I = np.zeros((nf, nc))
    for i in range(nc):
        I[2*i:2*i+3,i] = [1,2,1]
    return 0.5*I

def oned_injector(nf):
    """Simple projection-based restrictor from n-1 => floor(n/2)-1
    @param nf: number of fine grid points
    """
    n = nf+1
    nc = (n+1)//2 - 1
    J = np.zeros((nc,nf))
    for i in range(nc):
        J[i,2*i+1] = 1
    return J

def oned_full_weight(nf):
    """Full weighting restritor from n-1 => floor(n/2)-1
    @param nf: number of fine grid points
    """
    n = nf+1
    nc = (n+1)//2 - 1
    return oned_prolong(nc).T

# 1d interpolation operator
def interpolation_1d(nc, nf):
    """Simple interpolator from ceil(n/2)-1 => n-1 
    of nearby grid points
    @param nc: number of course grid points
    @param nf: number of fine grids. @nf not used since n is well defined with @nc
    """
    d = np.repeat([[1, 2, 1]], nc, axis=0).T
    I = np.zeros((3,nc), dtype=int)
    for i in range(nc):
        I[:,i] = [2*i, 2*i+1, 2*i+2]
    J = np.repeat([np.arange(nc)], 3, axis=0)
    P = sp.coo_matrix(
        (d.ravel(), (I.ravel(), J.ravel()))
        ).tocsr()
    return 0.5 * P

def restrictor_1d(nc, nf):
    return interpolation_1d(nc,nf).T

"""
AGM Prologation/Restricting Strategies
Based on (http://ftp.demec.ufpr.br/multigrid/Bibliografias/Briggs_et_al_2000_Tutorial_with_corrections.pdf)
chapter 8, AMG
"""
def dependence_matrix(A,theta=0.5):
    """ returns binary matrix of strong influences
        
    Parameters 
    ----------
    A : np.ndarray
        matrix to construct dependency matrix of
    theta : float
        threshold for determining strong dependence
        
    Return 
    --------
    S : np.ndarray
        binary matrix where s_ij=1 iff jth variable strongly 
    """
    assert(A.shape[0] == A.shape[1])
    n = A.shape[0]
    
    S = np.zeros((n,n), dtype=np.int)
    
    for i in range(n):
        
        # find ub (max) in row i (ith equation)
        lb = -A[i,0] if i>0 else -A[i,1]
        for k in range(n):
            if(i!=k):
                ub = max(lb,-A[i,k])
        
        # traverse through ith row to find strongly influencing vars (ignore 0s so no influence)
        for j in range(n):
            if(-A[i,j] >= theta * ub and A[i,j]!=0):
                S[i,j] = 1

    return S

def select_amg_coarse_grid(stencil,n,theta=0.5):
    """ Step 1 of coarse grid selection
    
    Parameters
    ----------
    stencil : numpy.ndarray
        2D stencil. Must be of len = 3
    n : int
        size of model problem
    theta : float
        threshold for dependency when determining strong influences
        
    Returns
    ----------
    c,f : np.ndarray's
        Set of initial coarse and fine grids
    """
    # construct initial dependency
    A = mg.twod_model_mat(stencil,n).toarray()
    N = A.shape[0]
    # element s_ij says "i depends on j" or "j influences i"
    dep_mat = dependence_matrix(A,theta)
    
    # course and fine grid points
    c = np.array([],dtype=np.int)
    f = np.array([],dtype=np.int)
    
    # number of elements that particular element influences (called "lambda" in tutorial)
    # acts as voting mechanism to decide who is selected for coarse
    influence_ct = np.array([sum(dep_mat[i,:]) for i in range(N)])
    
    while(N - len(c) - len(f) > 0):
        # get st(rongst) inf(luence) index 
        stinf_idx = np.argmax(influence_ct)
        
        # add active, strongeset influence to course grid
        c = np.append(c,stinf_idx)
        
        # remove strongest influence from active grid
        dep_mat[stinf_idx,:] = 0
        influence_ct[stinf_idx] = -1
        
        # gather active grid points that depend on stinf_idx and add to fine
        dep_idxs = np.nonzero(dep_mat[:,stinf_idx])
        f = np.append(f, dep_idxs)
        
        # remove new fine points from active grid and update values
        dep_mat[:,stinf_idx] = 0
        
        for dep_idx in dep_idxs:
            dep_mat[dep_idx,:] = 0; dep_mat[:,dep_idx] = 0
            influence_ct[dep_idx] = -1
        
        # for every newly added dependent point, increment their strong influencers
        for dep_idx in dep_idxs:
            
            # get all active influencers of dep_idx and increment their influencer ct
            new_inf_idxs = np.nonzero(dep_mat[:,dep_idx])
            for new_inf_idx in new_inf_idxs:
                influence_ct[new_inf_idx] += 1
                
    return c,f

def update_amg_coarse_grid(stencil,coarse_pts,fine_pts,n,theta=0.5):
    """ Step 2 of coarse grid selection
    
    Parameters
    ----------
    stencil : numpy.ndarray
        2D stencil. Must be of len = 3
    {coarse,fine}_pts : numpy.ndarray
        Current discrete set of coarse and fine grid points
    n : int
        size of model problem
    theta : float
        threshold for dependency when determining strong influences
        
    Returns
    ----------
    c,f : np.ndarray's
        Set of coarse and fine grids after satisfying rule (1)
    """
    # construct initial dependency
    A = mg.twod_model_mat(stencil,n).toarray()
    # element s_ij says "i depends on j" or "j influences i"
    dep_mat = dependence_matrix(A,theta)
    
    m = len(fine_pts)
    new_coarse_pts = np.array([],dtype=np.int)
    updated_coarse_pts = coarse_pts
    for i in range(m):
        for j in range(i+1,m):
            f_i,f_j = fine_pts[i],fine_pts[j]
            
            # only search for common coarse point if the fine points
            # have a strong connection
            if(A[f_i,f_j] or A[f_j,f_i]):
                common_inf = False
                for coarse_idx in updated_coarse_pts:
                    if(dep_mat[f_i,coarse_idx] and dep_mat[f_j,coarse_idx]):
                        common_inf = True
                        break
                    
                # if no common coarse point, move fine point i to coarse
                # we will not encounter index i ever again due to for loop ordering
                if(not common_inf):
                    new_coarse_pts = np.append(new_coarse_pts, fine_pts[i])
                    updated_coarse_pts = np.append(updated_coarse_pts, fine_pts[i])
                
    new_fine_pts = np.array([],dtype=np.int)      
    for fine_idx in fine_pts:
        if(fine_idx not in new_coarse_pts):
            new_fine_pts = np.append(new_fine_pts, fine_idx)
    
    return updated_coarse_pts, new_fine_pts

def amg_prolongation_operator(stencil,coarse_pts,fine_pts,n,theta=0.5):
    """ Constructs prologation operator from stencil and set of points
    
    Parameters
    ----------
    stencil : numpy.ndarray
        2D stencil. Must be of len = 3
    {coarse,fine}_pts : numpy.ndarray
        discrete set of coarse and fine grid points
    n : int
        size of model problem
    theta : float
        threshold for dependency when determining strong influences
        
    Returns
    ----------
    P : np.ndarray
        Prologation operator
    """
    
    # construct initial dependency
    A = mg.twod_model_mat(stencil,n).toarray()
    # element s_ij says "i depends on j" or "j influences i"
    dep_mat = dependence_matrix(A,theta)
    
    # number of variables
    N = A.shape[0]
    n_c,n_f = len(coarse_pts), len(fine_pts)
    assert(N == n_c + n_f)
    
    # construct prologation operator
    P = np.zeros((N,n_c))
    
    for i in range(N):
        # if coarse point, trivial
        if(i in coarse_pts):
            c_i = np.where(coarse_pts == i)
            P[i,c_i] = 1
        # if fine point, construct approximation
        else:
            """
            :C : coarse neighbors that influence i
            :F : fine neighbors that influence i
            :S : all other neighbors (coarse and fine)
            """
            C,F,S = [np.array([],dtype=np.int) for _ in range(3)]
            # neighbors of i
            neighbor_idxs = np.nonzero(A[i,:])
            # strong influencers of i
            influence_idxs = np.nonzero(dep_mat[i,:])
            
            for idx in neighbor_idxs:
                if(idx not in influence_idxs):
                    S = np.append(S,idx)
                elif(idx in coarse_pts):
                    C = np.append(C,idx)
                else:
                    F = np.append(F,idx)

            # construct denominator values
            coarse_idx_sum = np.array([ 
                np.sum(A[k,C]) for k in F], dtype=np.int
            )
            weak_idx_sum = np.sum( A[i,S] )
            denominator_val = A[i,i] + weak_idx_sum

            for j in range(n_c):
                c_j = coarse_pts[j]
                w_ij = A[i,c_j] + np.sum( [ 
                    A[i,k]*A[k,c_j] / coarse_idx_sum[k] for k in range(len(F))
                ] )
                w_ij = w_ij / denominator_val
                P[i,j] = -w_ij
                
    return P
