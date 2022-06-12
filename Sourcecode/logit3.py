import numpy as np 
import pandas as pd 

def q(theta, y, x): 
    '''q(theta,y,x): criterion function: L2 error'''
    assert theta.ndim == 1 
    assert x.ndim == 3 
    assert y.ndim == 2

    N,J,K = x.shape 
    assert (y.shape[0] == N) and (y.shape[1] == J), 'y and x do not conform '
    assert theta.size == K, 'x and theta do not conform'

    market_shares = choice_prob(theta, x) # (N,J)
    
    market_shares = np.clip(market_shares, 1e-8, 1-1e-8)

    loglike = (y * np.log(market_shares)).sum(axis=1)
    
    return -loglike # (N,) 

def starting_values(y, x): 
    assert x.ndim == 3 
    N,J,K = x.shape
    theta0 = np.zeros((K,))
    return theta0

def util(theta, x, MAXRESCALE=True): 
    '''util: compute the deterministic part of utility, v, and max-rescale it
    Args. 
        theta: (K,) vector of parameters 
        x: (N,J,K) matrix of covariates 
        MAXRESCALE (optional): bool, we max-rescale if True (the default)
    
    Returns
        v: (N,J) matrix of (deterministic) utility components
    '''
    assert theta.ndim == 1 
    N,J,K = x.shape 

    # deterministic utility 
    v = x @ theta # (N,J) 

    if MAXRESCALE: 
        # subtract the row-max from each observation
        v -= v.max(axis=1, keepdims=True)  # keepdims maintains the second dimension, (N,1), so broadcasting is successful
    
    return v 

def ccp_of_v(v): 
    # denominator 
    denom = np.exp(v).sum(axis=1, keepdims=True) # (N,1)
    
    # Conditional choice probabilites
    ccp = np.exp(v) / denom

    return ccp 

def choice_prob(theta, x):
    '''choice_prob(): Computes the (N,J) matrix of choice probabilities 
    Args. 
        theta: (K,) vector of parameters 
        x: (N,J,K) matrix of covariates 
    
    Returns
        ccp: (N,J) matrix of probabilities 
    '''
    assert theta.ndim == 1, f'theta should have ndim == 1, got {theta.ndim}'
    N, J, K = x.shape
    
    # deterministic utility 
    v = util(theta, x)
    
    # conditional choice probabilities 
    ccp = ccp_of_v(v)
    
    return ccp

def elast(theta, x, k, x_k_is_in_log=False): 
    N,J,K = x.shape
    ccp = choice_prob(theta, x)
    grad = dccp_dx(theta, x)[:,:,:,k]

    # elast = dy/dx * x/y
    if x_k_is_in_log: 
        # if our key var is log(p), then note that 
        # dy/dx = dy/dlog(x) * 1/x, 
        # so the 1/x eats the x 
        x_ = 1.0
    else:
        x_ = x[:,:,k].reshape(N,J,1)

    elast = grad * x_ / ccp.reshape(N,J,1)
    return elast

def dccp_dx(theta, x):
    '''(N,J,K) matrix of derivatives of CCPs wrt. covariates, x
    Returns 
        deriv: (N,J,K) matrix of partial derivatives     
    '''
    N,J,K = x.shape
    v = util(theta, x)
    ccp = ccp_of_v(v)

    # for each k, the partial deriv of x_k on v_j is 1 if j == k else 0
    # so a sequence of K diagonal matrices with theta[k] along the diagonal
    dv_dx = np.eye(J).reshape(J,J,1) * theta.reshape(1,1,K)
    
    # a common effect, working through the change in the denominator: 
    d_common = - np.sum(ccp.reshape(N,J,1,1) * dv_dx.reshape(1,J,J,K), axis=1, keepdims=True) # (N,1,J,K)
    
    # own-effect (the direct effect on v)
    d_own = dv_dx.reshape(1,J,J,K) 

    deriv = ccp.reshape(N,J,1,1) * (d_own + d_common)
        
    return deriv 

# analytic gradients wrt. parameters
def dccp_dtheta(theta, x): 
    '''dccp_dtheta(): (N,J,K) matrix of partial derivatives of CCP wrt. theta
    Returns: deriv (N,J,K) matrix of partial derivatives 

    TO TEST: 
        d = logit.dccp_dtheta(thetahat,x)
        k = 0
        assert thetahat[k] != 0.0, 'implement an absolute step'
        ccp0 = logit.choice_prob(thetahat, x)
        t2 = np.copy(thetahat)
        step = 1e-8*thetahat[k]
        t2[k] += step 
        ccp2 = logit.choice_prob(t2, x)
        dn = (ccp2-ccp0) / step 
        assert np.mean((d[:,:,k] - dn) ** 2) < 1e-8
    '''
    N,J,K = x.shape
    v = util(theta, x)
    ccp = (ccp_of_v(v)).reshape(N,J,1)
    
    dv_dtheta = x 
    d_common = (ccp*dv_dtheta).sum(1,keepdims=True)
    deriv = ccp * (dv_dtheta - d_common)
    
    return deriv

def dq(theta,y,x): 
    '''Gradient vector of the criterion function wrt. parameters
    Returns:
        grad: (N,K) matrix of derivatives of the criterion function, q
    ''' 
    N,J,K = x.shape
    assert y.shape[0] == N 
    assert y.shape[1] == J 
    
    ms = choice_prob(theta, x)

    ms = np.clip(ms, 1e-8, 1.0-1e-8)
        
    dccp = dccp_dtheta(theta, x)

    yms = (y/ms).reshape(N,J,1) # add 3rd dimension to allow broadcasting 
    dll = yms*dccp # elementwise product, expanding yms to (N,J,K)
    
    # sum over the J-dimension
    grad = -dll.sum(axis=1)
    
    return grad

def ddq(theta,y,x): 
    '''Hessian, by the outer product of the scores'''
    N,J,K = x.shape
    s = dq(theta,y,x)
    H = s.T@s/N
    return H 

def test_dccp_dx(theta, x, k, j, tol=1e-8, h=1e-6):
    '''Test the function dccp_dx()'''
    N,J,K = x.shape
    assert not np.any(x[:,j,k] == 0.0), 'if there are zeros in x, we have to use an absolute step'

    d = dccp_dx(theta, x)

    ccp0 = choice_prob(theta, x)

    # take a forward step 
    x2 = np.copy(x)
    step = h*x[:,j,k]
    x2[:,j,k] += step

    # evaluate at new step 
    ccp2 = choice_prob(theta, x2)

    # compute finite difference 
    dn = (ccp2-ccp0) / step.reshape(N,1)

    norm = np.mean((d[:,:,j,k] - dn)**2)
    print(f'norm of analytic - numeric deriv: {norm:12.5g}')
    return norm < tol


def elasticity_numeric(theta, x, k=3, h=1e-2): 
    '''Compute numerical elasticity'''

    N,J,K = x.shape

    print(f'Computing elasticity for variable {k}: assuming that it is in log!')

    ccp = choice_prob(theta, x)

    # initialize
    E_own   = np.zeros((N, J))
    E_cross = np.zeros((N, J))
    dpdx    = np.zeros((N, J))

    for j in range(J):
        # Copy
        x2 = x.copy()
        h = 0.01 # relative step of 1% 
        x2[:, j, k] += h # adding 0.01 to the log is almost like adding 1% 

        ccp2 = choice_prob(theta, x2)
        dccp = ccp2 - ccp

        x0 = x[:, j, k]

        # Own price elasticity
        dy = dccp[:, j]
        y0 = ccp[:, j]
        E_own[:, j] = dy/y0 / h #utilizing that the variable is in log, dy/dx x/y = dy/dlogx * 1/y

        # Cross price elasticity
        other_cars = [k for k in range(J) if k != j]
        dy = dccp[:, other_cars]
        y0 = ccp[:, other_cars]
        E_cross[:, j] = np.mean(dy/y0, axis=1) / h 

    print(f'Own-price elasticity: {E_own.mean().round(8)}')

    print(f'Cross-price elasticity: {E_cross.mean().round(8)}')
