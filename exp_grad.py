import numpy as np

def ext_x(x):
    return np.concatenate((np.ones(shape = (x.shape[0], 1)), x), axis = 1)

def get_lam(eta):
    lam = np.exp(eta)
    return lam / lam[1:].sum()



def exponentiated_gradient(str_x, str_y, str_p, ttr_x, ttr_p, 
                                     eps = 1e-3, tol = 2e-3, B = 5, lr = 4e-3, 
                                     max_iter = 4000, verbose = 0, reg = 1e-5):
    """
    str_x, str_y, str_p: x, y, proababilites from source training dataset
    ttr_x, ttr_p: x and probabilities from target training dataset
    eps: constraint relaxation
    tol: tolerance levels for convergence
    B: maximum value for Lagrange multiplier
    lr: learning rate
    reg: regularizer strength
    """

    d = str_x.shape[1] + 1
    beta = np.zeros(shape = (d, 2))
    eta = np.array([0, 0, 0], dtype = "float64")

    str_x_ext = ext_x(str_x)
    ttr_x_ext = ext_x(ttr_x)

    err = 1.
    ITER = 0
    
    while ITER < max_iter and err > tol:
    
        str_w = np.exp(str_x_ext @ beta)
        ttr_w = np.exp(ttr_x_ext @ beta)
        
        str_u = str_w * str_p 
        ttr_u = ttr_w * ttr_p 
        
        lam = B * get_lam(eta)
        
        grad_beta = - ((ttr_u / ttr_u.sum(axis = 1, keepdims = True))[:, :, None] * ttr_x_ext[:, None, :]).mean(axis = 0).T
        grad_beta += (lam[1] - lam[2]) * (str_u[:, :, None] * str_x_ext[:, None, :]).mean(axis = 0).T
        beta = (1 - lr * reg) * beta - lr * grad_beta
        
        constraint_function = np.log(str_u.mean(axis = 0).sum())
        err = np.linalg.norm(lr * grad_beta) / np.linalg.norm(beta) + max(np.abs(constraint_function), eps) 
        
        eta[1] += lr * (constraint_function if np.exp(constraint_function) -1 > eps else 0)
        eta[2] += lr * (- constraint_function if np.exp(constraint_function) -1 < -eps else 0)
        
        if verbose > 0 and ITER % 10 == 0:
            print("ITER",  ITER, "error: ", err, "lambda", lam, "constr", constraint_function)
        ITER += 1
        
    if ITER >= max_iter:
        print("\nTerminated without convergence")
        
    return beta


def profile_likelihood_opt(str_x, str_y, str_p, ttr_x, ttr_p, 
                        tol = 2e-3, lr = 4e-3, max_iter = 4000, 
                        verbose = 0, reg = 1e-5):
    
    """
    str_x, str_y, str_p: x, y, proababilites from source training dataset
    ttr_x, ttr_p: x and probabilities from target training dataset
    tol: tolerance levels for convergence
    lr: learning rate
    """
    
    d = str_x.shape[1] + 1
    beta = np.zeros(shape = (d, 2))

    str_x_ext = ext_x(str_x)
    ttr_x_ext = ext_x(ttr_x)
    
    n_1 = str_x_ext.shape[0]
    n_0 = str_x_ext.shape[0]
    n = n_0 + n_1

    err = 1.
    ITER = 0
    
    while ITER < max_iter and err > tol:
    
        str_w = np.exp(str_x_ext @ beta)
        ttr_w = np.exp(ttr_x_ext @ beta)
        
        str_u = str_w * str_p 
        ttr_u = ttr_w * ttr_p 
        
        grad_beta = ((ttr_u / ttr_u.sum(axis = 1, keepdims = True))[:, :, None] * ttr_x_ext[:, None, :]).sum(axis = 0).T
        grad_beta -= ((ttr_u / (ttr_u.sum(axis = 1, keepdims = True) + n_1 / n_0))[:, :, None] * ttr_x_ext[:, None, :]).sum(axis = 0).T
        grad_beta -= ((str_u / (str_u.sum(axis = 1, keepdims = True) + n_1 / n_0))[:, :, None] * str_x_ext[:, None, :]).sum(axis = 0).T
        
        beta = (1 - lr * reg) * beta + lr * grad_beta / n
        err = np.linalg.norm(lr * grad_beta/n) / np.linalg.norm(beta) 
        
        if verbose > 0 and ITER % 10 == 0:
            print("ITER",  ITER, "error: ", err, )
        ITER += 1
        
    if ITER >= max_iter:
        print("\nTerminated without convergence")
        
    return beta