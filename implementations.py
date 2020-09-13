import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from functions import *



#-----------------------------------------------------------------------------------------------------------------------------------
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    ''' 
    least squares gradient descent
    INPUT:
        y: labels
        tx: features
        initial_w: initial weights
        max_iters: maximum number of iterations
        gamma: learning rate
    '''
    w=initial_w
    losses = []
    ws=[initial_w]
    for n_iter in range(max_iters):
        # compute gradient and loss
        loss = compute_mse(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        # update w by gradient
        w = w - gamma * gradient

        # store w and loss  
        losses.append(loss)
        ws.append(w)
    return ws[-1], losses[-1]
#-----------------------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------------------
def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    ''' 
    stochastic gradient descent
    INPUT:
        y: labels
        tx: features
        initial_w: initial weights
        max_iters: maximum number of iterations
        batch_size: unused (for signature)
        gamma: learning rate
    '''
     #Initialize list of weights and losses
    ws = [initial_w]
    losses = []
    w = initial_w
    #For each minibatch


    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        #For each iteration
        
            # compute loss 
            loss = compute_mse(minibatch_y, minibatch_tx, w)
            # compute gradient
            grad = compute_gradient(minibatch_y, minibatch_tx, w)
            
            # update w by gradient  
            w = w - gamma * grad

            # store w and loss
            ws.append(w)
            losses.append(loss)
    return  ws[-1], losses[-1]
#-----------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------
def least_squares(y, tx):
    '''
    INPUT:
        y: labels
        tx: features
    '''
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    mse = compute_mse(y, tx, w)
    return w, mse

def ridge_regression(y, tx, lambda_):
    '''
    INPUT:
        y: labels
        tx: features
        lambda_: step size
    '''
    w = np.linalg.solve(tx.T.dot(tx) + 2*lambda_ *
                        np.identity(tx.shape[1])*y.shape[0], tx.T.dot(y))
    mse = compute_mse(y, tx, w)
    return w, mse
#-----------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    '''
    INPUT:
        y: labels
        tx: features
        initial_w: initial weights
        max_iters: maximum number of iterations
        gamma: learning rate
    '''
    w = initial_w
    for iter in range(max_iters):
        loss = calculate_loss_Log(y, tx, w)
        # compute the gradient:
        grad = calculate_gradient_Log(y, tx, w)
        # update w:
        w = w - gamma*grad
    return w, loss
#-----------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    '''
    INPUT:
        y: labels
        tx: features
        initial_w: initial weights
        max_iters: maximum number of iterations
        gamma: learning rate
        lambda_: step size
    '''
    w = initial_w
    for iter in range(max_iters):
        # update w:
        loss = calculate_loss_reg_Log(y, tx, w,lambda_)
        w = w - gamma * penalized_gradient(y, tx, w, lambda_)
    return w, loss