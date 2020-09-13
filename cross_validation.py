#-----------------------------------------------------------------------------------------
def cross_validation_RegLogGD(y,x,k,seed,lambda_, initial_w, max_iters=10000, gamma=1):
    """cross validation using Regularised logistic gradient descent"""
    def estimator(arg1,arg2): 
        return reg_logistic_regression(arg1,arg2, lambda_, initial_w, max_iters, gamma)
    return cross_validation_functional(y, x, k, seed,estimator)


def ridge_regression_find_best_lambda(y, tx, k_fold, lambdas, seed):
    """
    Ridge regression used with cross validation technique as well as finding best lambdas by lowest test RMSE.
    Returns best weights and lambda.
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    weights = []
    rmse_tr = []
    rmse_te = []
    
    #go through all lambdas
    for lambda_ in lambdas:   
        w_temp = []
        rmse_train_temp = []
        rmse_test_temp = []
        for i in range(k_fold):
            
            #get test and train data
            testy, trainy = y[k_indices[i]], y[np.delete(k_indices,i,0).flatten()]
            txtest, txtrain = tx[k_indices[i]], tx[np.delete(k_indices,i,0).flatten()]
            
            #get weights and errors of a fold
            w = ridge_regression(trainy, txtrain, lambda_)
            rmse_train = np.sqrt(2 * compute_mse(trainy, txtrain, w))
            rmse_test = np.sqrt(2 * compute_mse(testy, txtest, w))
            w_temp.append(w)
            rmse_train_temp.append(rmse_train)
            rmse_test_temp.append(rmse_test)
            
        #get average values from cross-validation  
        weights.append(np.mean(w_temp, axis = 0))
        rmse_tr.append(np.mean(rmse_train_temp))
        rmse_te.append(np.mean(rmse_test_temp))
        
    #get best lambda
    best_rmse = np.min(rmse_te)
    best_lambda = lambdas[np.argmin(rmse_te)]
    best_w = weights[np.argmin(rmse_te)]
    return best_w, best_lambda


def cross_validation_least_squares(y, tx, k, seed):
    """
    Least squares used with cross validation technique.
    Returns average weight, average train RMSE, average test RMSE.
    """
    weights = []
    rmses_te = []
    rmses_tr = []
    # split data in k fold
    k_indices = build_k_indices(y, k, seed)
    numk=k_indices.shape[0]
    for k in range(numk):
        #get test and train data
        testy, trainy = y[k_indices[k]], y[np.delete(k_indices,k,0).flatten()]
        txtest, txtrain = tx[k_indices[k]], tx[np.delete(k_indices,k,0).flatten()]
        
        #get weights and errors of a fold
        w, _ = least_squares(trainy, txtrain)
        mseTraining, mseTest = compute_mse(trainy,txtrain,w), compute_mse(testy,txtest,w)
        rmse_tr, rmse_te = np.sqrt(2*mseTraining), np.sqrt(2*mseTest)
        
        weights.append(w)
        rmses_tr.append(rmse_tr)
        rmses_te.append(rmse_te)
    avgw = np.mean(weights, axis = 0)
    avgrmse_te = np.mean(rmse_te)
    avgrmse_tr = np.mean(rmse_tr)
    return avgw, avgrmse_tr, avgrmse_te

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    testy, trainy = y[k_indices[k]], y[np.delete(k_indices, k, 0).flatten()]
    tx = build_poly(x, degree)
    txtest, txtrain = tx[k_indices[k]
                         ], tx[np.delete(k_indices, k, 0).flatten()]
    w = ridge_regression(trainy, txtrain, lambda_)

    mseTest = compute_mse(testy, txtest, w)
    mseTraining = compute_mse(trainy, txtrain, w)
    rmse_tr, rmse_te = np.sqrt(2*mseTraining), np.sqrt(2*mseTest)
    loss_tr, loss_te = rmse_tr, rmse_te
    return w, loss_tr, loss_te

def cross_validationf(y, x, k_indices, lambda_, degree):
    avgw, avgltr, avglte = 0, 0, 0
    numk = k_indices.shape[0]
    for k in range(numk):
        w, ltr, lte = cross_validation(y, x, k_indices, k, lambda_, degree)
        avgw, avgltr, avglte = avgw + w/numk, avgltr + ltr/numk, avglte+lte/numk
    return avgw, avgltr, avglte

def cross_validation_find_best_degree(y, x, k_fold, lambdas, degrees, seed):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    best_rmses = []
    best_lambdas = []
    for degree in degrees:
        rmse_te = []
        for lambda_ in lambdas:
            rmse_test_temp = []
            for i in range(k_fold):
                _, _, rmse_test = cross_validation(
                    y, x, k_indices, i, lambda_, degree)
                rmse_test_temp.append(rmse_test)
            rmse_te.append(np.mean(rmse_test_temp))
        best_rmses.append(np.min(rmse_te))
        best_lambdas.append(lambdas[np.argmin(rmse_te)])

    thermse = np.min(best_rmses)
    thelambda = best_lambdas[np.argmin(best_rmses)]
    index = np.where(best_rmses == thermse)
    thedegree = int(degrees[index])
    #print("Best degree={d}, Best lambda={l:.8f}, Best test RMSE={tr:.3f}".format(d=thedegree, l=thelambda, tr=thermse))
    return thermse, thelambda, thedegree


def cross_validation_find_best(y, x, k_fold,seed, lambdas, gammas, initial_w, max_iters=10000):
    """Function that gets the best parameters for logistic regression through cross validation"""
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    best_accuracy = 0
    best_lambda = 0
    best_gamma = 0
    best_prediction= []
    for gamma in gammas:
        for lambda_ in lambdas:
            currentw, currentaccuracy ,currentprediction =cross_validation_RegLogGD(y,x,k_fold,seed,lambda_, initial_w, max_iters, gamma)
            if currentaccuracy> best_accuracy: 
                best_accuracy= currentaccuracy
                best_lambda =lambda_
                best_gamma=gamma
                best_prediction[0] = currentprediction
    print("Best accuracy={ba}, Best lambda={bl:.8f}, Best gamma={bg:.3f}".format(ba=best_accuracy, bl=best_lambda, bg=best_gamma))
    return best_accuracy, best_lambda, best_gamma
#-----------------------------------------------------------------------------------------