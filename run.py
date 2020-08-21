import numpy as np
import matplotlib.pyplot as plt
from functions import *
from implementations import *



from proj1_helpers import *
DATA_TRAIN_PATH = 'train.csv' 
#DATA_TRAIN_PATH = r'C:\Users\lsyvi\Documents\Master\ML_course-master\projects\project1\data\train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

tX[tX==-999]=0
tX,_,_=standardize(tX)
tx = np.c_[np.ones((tX.shape[0], 1)), tX]
initial_w=np.zeros(tx.shape[1])

#...................................................................................................
print('------------------------------gradient descent-------------------------------')
max_iters=200
gamma=1e-1
initial_w=np.zeros(tx.shape[1])
w_GD,loss_GD=least_squares_GD(y, tx, initial_w, max_iters, gamma)
print('loss :',loss_GD)
label_GD=predict_label(tx,w_GD)
print('accuracy :',accuracy(label_GD,y))
#...................................................................................................
print('------------------------------stochastic gradient descent-------------------------------')
max_iters = 700
gamma = 1e-1
batch_size=10
initial_w = np.zeros(tx.shape[1])
w_SGD,loss_SGD = least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma)
print('loss :',loss_SGD)
label_SGD = predict_label(tx,w_SGD)
print('accuracy :',accuracy(label_SGD,y))
#...................................................................................................
print('------------------------------least squares -------------------------------')
max_iters=100
gamma=2e-2
w_least_squares,loss_least_squares=least_squares(y, tx)
print('loss :',loss_least_squares)
label_least_squares=predict_label(tx,w_least_squares)
print('accuracy :',accuracy(label_least_squares,y))
#...................................................................................................
print('------------------------------ridge regression-------------------------------')
max_iters=100
gamma=1e-1
lambda_=1e-1
w_ridge,loss_ridge=ridge_regression(y, tx, lambda_)
print('loss :',loss_ridge)
label_ridge=predict_label(tx,w_ridge)
print('accuracy :',accuracy(label_ridge,y))
#...................................................................................................
print('------------------------------logistic regression-------------------------------')
max_iters=100
gamma=2e-2
w_logistic,loss_logistic=logistic_regression(y, tx, initial_w, max_iters, gamma)
print('loss :',loss_logistic)
label_logistic=predict_label(tx,w_logistic)
print('accuracy :',accuracy(label_logistic,y))
#...................................................................................................
print('------------------------------regularized logistic regression-------------------------------')
max_iters=100
gamma=2e-2
lambda_=1e-1
w_reg,loss_reg=reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
print('loss :',loss_reg)
label_reg=predict_label(tx,w_reg)
print('accuracy :',accuracy(label_reg,y))
#...................................................................................................
print('------------------------------least squares with feature augmentation -------------------------------')
max_iters=100
gamma=2e-2
augmented_tx=build_polyExp(tX,9)
w_augmented_least_squares,loss_augmented_least_squares=least_squares(y, augmented_tx)
print('loss :',loss_augmented_least_squares)
label_least_squares=predict_label(augmented_tx,w_augmented_least_squares)
print('accuracy :',accuracy(label_least_squares,y))



DATA_TEST_PATH = 'test.csv' 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

tX_test[tX_test==-999]=0
tX_test,_,_=standardize(tX_test)
tX_test=build_polyExp(tX_test,9)

OUTPUT_PATH = 'final_submission.csv' 
y_pred = predict_labels(w_augmented_least_squares, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)














