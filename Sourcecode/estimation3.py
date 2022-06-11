import numpy as np
import pandas as pd 
from scipy import optimize
import time 

def estimate_m(model, y, x, method='BFGS', cov_type='Outer Product', **kwargs): 
    '''thetahat, se = estimate_m()''' 
    theta0 = model.starting_values(y,x)

    if method != 'BHHH': 
        r = estimate(model.q, theta0, y, x, cov_type=cov_type, method=method, **kwargs)
    elif method == 'BHHH': 
        assert getattr(model, 'dq', None) is not None, f'If BHHH is requested, model must have the function "dq"'
        jac = lambda theta : model.dq(theta, y, x)   # analytical gradient 
        hess = lambda theta : model.ddq(theta, y, x) # outer product of the scores approximation 
        r = estimate(model.q, theta0, y, x, cov_type=cov_type, method='CG-Newton', 
                     jac=jac, hess=hess, **kwargs)
    else: 
        raise Exception(f'Method, {method}, not allowed')

    if 'se' not in r:
        se = None
    else:
        se = r['se']

    return r['theta_hat'], se

def estimate(
        q, 
        theta0: np.ndarray, 
        y: np.ndarray, 
        x: np.ndarray, 
        cov_type='Outer Product',
        options = {'disp': True},
        **kwargs
    ):
    """Takes a function and returns the minimum, given start values and 
    variables to calculate the residuals.

    Args:
        q: The function to minimize. Must return an (N,) vector.
        theta0 (list): A list with starting values.
        y (np.array): Array of dependent variable.
        x (np.array): Array of independent variables.
        cov_type (str, optional): String for which type of variances to 
        calculate. Defaults to 'Outer Product'. If =None, no calculations
        options: dictionary with options for the optimizer (e.g. disp=True,
        which tells it to display information at termination.)

    Returns:
        dict: Returns a dictionary with results from the estimation.
    """
    assert callable(q), 'q must be a callable function'
    
    # The minimizer can't handle 2-D arrays, so we flatten them to 1-D.
    theta0 = theta0.flatten()
    N = y.size

    # The objective function is the average of q(), 
    # but Q is only a function of one variable, theta, 
    # which is what minimize() will expect
    Q = lambda theta: np.mean(q(theta, y, x))

    # call optimizer
    result = optimize.minimize(
        Q, theta0, options=options, **kwargs
        )

    # collect output in a dict 
    res = {
        'theta_hat': result.x,
        'success':  result.success, # bool, whether convergence was succesful 
        'nit':      result.nit, # no. algorithm iterations 
        'nfev':     result.nfev, # no. function evaluations 
        'fun':      result.fun # function value at termination 
    }

    # if cov_type == None, we do not calculate the variance 
    if cov_type is not None: 

        try: 
            # compute standard errors 
            cov, se = variance(q, y, x, thetahat=result.x, cov_type=cov_type)   

            # add to results 
            res['se']       = se
            res['t'] = result.x / se
            res['cov']      = cov

        except: 
            print(f'Error! Unable to compute variance.')

    return res

def variance(
        q, # function taking three inputs: q(theta, y, x) 
        y: np.ndarray, 
        x: np.ndarray, 
        thetahat: np.ndarray,
        cov_type: str,
        hess_inv=None 
    ):
    """Calculates the variance for the likelihood function.

    Args:
        >> q: Function taking three inputs, q(theta,y,x), where we minimize the mean of q.
        >> y (np.ndarray): Dependent variable.
        >> x (np.ndarray): Independent variables.
        >> thetahat: parameter estimates 
        >> cov_type (str): Type of calculation to use in estimation.
        >> hess_inv: inverse of the Hessian (minimize sometimes returns an estimate hereof, 
            although it is not advisable to use it.)

    Returns:
        tuple: Returns the variance-covariance matrix and standard errors.
    """
    assert callable(q), 'q must be a callable function'
    N = y.size
    P = thetahat.size

    f_q = lambda theta : q(theta,y,x) # as q(), but only a function of theta, whereas q also takes y,x as inputs
    
    if cov_type in ['Outer Product', 'Sandwich']:    
        # numerical gradients are needed
        s = centered_grad(f_q, thetahat)
        
        # "B" matrix: outer product of the scores / N
        B = (s.T@s)/N

    if (hess_inv is None) and (cov_type in ['Hessian', 'Sandwich']): 
        # Hessian needed
        H = hessian(f_q, thetahat)/N # divide by N since the function merely sums 
        hess_inv = np.linalg.inv(H)

    
    
    # cov: P*P covariance matrix of theta 
    if cov_type == 'Hessian':
        A_inv = hess_inv
        cov = 1/N * A_inv
    elif cov_type == 'Outer Product':
        cov = 1/N * np.linalg.inv(B)
    elif cov_type == 'Sandwich':
        A_inv = hess_inv
        cov = 1/N * (A_inv @ B @ A_inv)

    # se: P-vector of std.errs. 
    se = np.sqrt(np.diag(cov))

    return cov, se


def centered_grad(f, x0: np.ndarray, h:float=1.49e-08):
    '''centered_grad: numerical gradient calculator
    Args.
        f: function handle taking *one* input, f(x0). f can return a vector. 
        x0: P-vector, the point at which to compute the numerical gradient 

    Returns
        grad: N*P matrix of numericalgradients. 
    '''
    assert callable(f), 'f must be a callable function'
    assert x0.ndim == 1, f'Assumes x0 is a flattened array'
    P = x0.size 

    # evaluate f at baseline 
    f0 = f(x0)
    N = f0.size

    # intialize output 
    grad = np.zeros((N, P))
    for i in range(P): 

        # initialize the step vectors 
        x1 = x0.copy()  # forward point
        x_1 = x0.copy() # backwards 

        # take the step for the i'th coordinate only 
        if x0[i] != 0: 
            x1[i] = x0[i]*(1.0 + h)  
            x_1[i] = x0[i]*(1.0 - h)
        else:
            # if x0[i] == 0, we cannot compute a relative step change, 
            # so we just take an absolute step 
            x1[i] = h
            x_1[i] = -h
        
        step = x1[i] - x_1[i] # the length of the step we took 
        grad[:, i] = ((f(x1) - f(x_1))/step).flatten()

    return grad



def hessian( fhandle , x0 , h=1e-5 ): 
    '''hessian(): computes the (K,K) matrix of 2nd partial derivatives
        using the aggregation "sum" (i.e. consider dividing by N)

    Args: 
        fhandle: callable function handle, returning an (N,) vector or scalar
        x0: K-array of parameters at which to evaluate the derivative 

    Returns: 
        hess: (K,K) matrix of second partial derivatives 
    
    Example: 
        from scipy.optimize import rosen, rosen_der, rosen_hess
        > x0 = np.array([-1., -4.])
        > rosen_hess(x0) - estimation.hessian(rosen, x0)
        The default step size of h=1e-5 gives the closest value 
        to the true Hessian for the Rosenbrock function at [-1, -4]. 
    '''

    # Computes the hessian of the input function at the point x0 
    assert x0.ndim == 1 , f'x0 must be 1–dimensional'
    assert callable(fhandle), 'fhandle must be a callable function handle'

    # aggregate rows with a raw sum (as opposed to the mean)
    agg_fun = np.sum

    # Initialization
    K = x0.size
    f2 = np.zeros((K,K)) # double step
    f1 = np.zeros((K,))  # single step
    h_rel = h # optimal step size is smaller than for gradient
                
    # Step size 
    dh = np.empty((K,))
    for k in range(K): 
        if x0[k] == 0.0: # use absolute step when relative is impossible 
            dh[k] = h_rel 
        else: # use a relative step 
            dh[k] = h_rel*x0[k]

    # Initial point 
    time0 = time.time()
    f0 = agg_fun(fhandle(x0)) 
    time1 = time.time()

    # expected time until calculations are done 
    sec_per_eval = time1-time0 
    evals = K + K*(K+1)//2 
    tot_time_secs = sec_per_eval * evals 
    if tot_time_secs > 5.0: # if we are slow, provide an ETA for the user 
        print(f'Computing numerical Hessian, expect approx. {tot_time_secs:5.2f} seconds (for {evals} criterion evaluations)')

    # Evaluate single forward steps
    for k in range(K): 
        x1 = np.copy(x0) 
        x1[k] = x0[k] + dh[k] 
        f1[k] = agg_fun(fhandle(x1))

    # Double forward steps
    for k in range(K): 
        for j in range(k+1): # only loop to the diagonal!! This is imposing symmetry to save computations
            
            # 1. find the new point (after double-stepping) 
            x2 = np.copy(x0) 
            if k==j: # diagonal steps: only k'th entry is changed, taking two steps 
                x2[k] = x0[k] + dh[k] + dh[k] 
            else:  # we have taken both a step in the k'th and one in the j'th directions 
                x2[k] = x0[k] + dh[k] 
                x2[j] = x0[j] + dh[j]  

            # 2. compute function value 
            f2[k,j] = agg_fun(fhandle(x2))
            
            # 3. fill out above the diagonal ()
            if j < k: # impose symmetry  
                f2[j,k] = f2[k,j]

    hess = np.empty((K,K))
    for k in range(K): 
        for j in range(K): 
            hess[k,j] = ((f2[k,j] - f1[k]) - (f1[j] - f0)) / (dh[k] * dh[j])

    return hess 
    

def print_table(
        theta_label: list,
        results: dict,
        headers:list = ["", "Beta", "Se", "t-values"],
        title:str = "Results",
        num_decimals:int = 4
    ):
    """Prints a nice looking table, must at least have coefficients, 
    standard errors and t-values. The number of coefficients must be the
    same length as the labels.

    Args:
        theta_label (list): List of labels for estimated parameters
        results (dict): The output from estimate()
        dictionary with at least the following keys:
            'theta_hat', 'se', 't_values'
        headers (list of 4 strings, optional): Column headers. Defaults to 
            ["", "Beta", "Se", "t-values"].
        title (str, optional): Table title. Defaults to "Results".
        num_decimals: (int) where to round off results (=None to disable)
    """
    assert len(theta_label) == len(results['theta_hat'])
    
    tab = pd.DataFrame({
       'theta': results['theta_hat'], 
        'se': results['se'], 
        't': results['t']
        }, index=theta_label)
    
    if num_decimals is not None: 
        tab = tab.round(num_decimals)
    
    # Print the table
    opt_outcome = 'succeeded' if results['success'] else 'failed'
    print(f'Optimizer {opt_outcome} after {results["nit"]} iter. ({results["nfev"]} func. evals.). Final criterion: {results["fun"]: 8.4g}.')
    print(title)
    return tab 

def test_analytic_gradient(model, theta, y, x, tol=1e-6, h=1e-7): 
    assert getattr(model, 'dq', None) is not None, f'model has no function, "dq"'
    q  = lambda theta : model.q(theta, y, x)
    g_a = model.dq(theta, y, x)
    g_n = centered_grad(q, theta, h=h)

    print(f'Analytic mean abs: {np.mean(np.abs(g_a))} numeric = {np.mean(np.abs(g_n))}')
    print(f'Mean abs diff: {np.mean(np.abs(g_a - g_n))}')

    return np.mean(np.abs(g_a - g_n)) < tol
