import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    x2 = (x**0).reshape([x.shape[0], 1])
    for i in range(1, degree+1):
        x2 = np.concatenate([x2, np.power(x.reshape([x.shape[0], 1]), i)], axis=1)
    return x2




def build_polyExp(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    x2 = (x[:,0].reshape([x.shape[0], 1]))**0
    for i in range(1, degree+1):
        x2 = np.concatenate([x2, np.power(x, i)], axis=1)
    return x2


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse



def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


def split_data(x, y, ratio, seed=0):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    indices = np.random.permutation(x.shape[0])
    num = x.shape[0]
    n = int(np.round(num*ratio))
    trainingx, trainingy = x[indices[0:n]], y[indices[0:n]]
    testx, testy = x[indices[n:]], y[indices[n:]]

    return trainingx, trainingy, testx, testy


def build_k_indices(y, k_fold, seed=0):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)













def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")


def sigmoid(t):
    """apply sigmoid function on t."""
    return 1 / (1 + np.exp(-1*t))




def calculate_gradient_Log(y, tx, w):
    """compute the gradient of loss."""
    b=sigmoid(tx.dot(w))
    return tx.T.dot(b - y)




def predict_label(tx,w):
    predictions=sigmoid(tx.dot(w))
    # print('real value',predictions,'\n')
    predictions[predictions<0.5]=-1
    predictions[predictions>=0.5]=1
    # print('set to 1 or -1',predictions,'\n')
    return predictions

def accuracy(labels, predicted_labels):
    return np.sum(labels==predicted_labels)/labels.shape[0]




def calculate_hessian_Log(y, tx, w):
    """return the hessian of the loss function."""
    b=sigmoid(tx.dot(w))
    c=np.ones([tx.shape[0], 1]) - b
    z=b*(c)
    S = np.diagflat(z)
    return tx.T.dot(S.dot(tx))


def logistic_regression_helper_newton(y, tx, w):
    """return the loss, gradient, and hessian."""
    # return loss, gradient, and hessian: TODO
    loss = calculate_loss_Log(y, tx, w)
    grad = calculate_gradient_Log(y, tx, w)
    hess = calculate_hessian_Log(y, tx, w)
    return loss, grad, hess

def logistic_regression_helper(y, tx, w):
    """return the loss, gradient, and hessian."""
    # return loss, gradient, and hessian: TODO
    loss = calculate_loss_Log(y, tx, w)
    grad = calculate_gradient_Log(y, tx, w)
    return loss, grad

def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    # return loss, gradient and hessian:
    loss, grad, hess = logistic_regression_helper(y, tx, w)
    # update w:
    w = w - gamma*np.linalg.inv(hess).dot(grad)
    return loss, w


def penalized_logistic_regression_helper_newton(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    # return loss, gradient, and hessian
    loss, grad, hess = logistic_regression_helper(y, tx, w)
    return loss + lambda_/2 * w.T.dot(w), grad + lambda_*w, hess + 1*lambda_*np.identity(w.shape[0])



def penalized_gradient(y, tx, w, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    # return loss, gradient:
    grad = calculate_gradient_Log(y, tx, w)
    return grad + lambda_*w


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    error = y - tx.dot(w)
    gradient = -tx.T.dot(error) / len(error)
    return gradient


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    threshold=0.8
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)            
            w = w - gamma * gradient
            temp_labels = predict_label(tx,w)
            accuracy_=accuracy(y,temp_labels)
            if n_iter % 10 == 0:
                print("Current iteration={i}, accuracy={a}".format(i=n_iter, a=accuracy_))
            if accuracy_>threshold:
                break
    return  w, temp_labels



def stochastic_logistic_regression(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    w = initial_w
    temp_labels=predict_label(tx,w)
    accuracy_=accuracy(y,temp_labels)
    print(accuracy_)
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # compute the gradient:
            grad = calculate_gradient_Log(minibatch_y, minibatch_tx, w)
            # update w:
            w = w - gamma*grad
            temp_labels=predict_label(tx,w)
            accuracy_=accuracy(y,temp_labels)
            if n_iter % 10 == 0:
                print("Current iteration={i}, accuracy={a}".format(i=iter, a=accuracy_))
    return w, temp_labels

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True,seed=0):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    np.random.seed(seed)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

            






def cross_validation_functional(y, tx, k, seed,estimator):
    """Functional form of cross validation"""
    k_indices = build_k_indices(y, k, seed)
    weights =[ ]
    numk=k_indices.shape[0]
    for k in range(numk):
        testy, trainy = y[k_indices[k]], y[np.delete(k_indices,k,0).flatten()]
        txtest, txtrain = tx[k_indices[k]], tx[np.delete(k_indices,k,0).flatten()]

        w, temp_labels = estimator(trainy, txtrain)


        accuracy_=accuracy(y,temp_labels)
        print("k={k},accuracy={a}".format(a=accuracy_,k=k)) 
        weights.append(w)
    avgw = np.mean(weights,axis =0)
    return avgw, accuracy_, predict_label(tx,avgw)




