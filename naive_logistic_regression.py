
"""
naive_logistic_regression.py is licensed under the
GNU General Public License v3.0
Permissions of this strong copyleft license are conditioned on making available complete source code of licensed works and modifications, which include larger works using a licensed work, under the same license. Copyright and license notices must be preserved. Contributors provide an express grant of patent rights.
"""

from mathplus import *

# Demo of simplified logistic regression using mathplus

# this is a naive implementation of logistic regression
# that does not use cross entropy -(y.log(h(x)) + (1-y)*log(1-h(x)))
# where h(x) is the predicted result (h = hypothesis)
# instead the simplistic error function (predictions-y) is used


X = array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = array([0, 0, 1, 1])


# the hypothesis is calculated using the sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
    

# create a vectorized function    
sigmoid = Utils.make_vfunc(sigmoid)


# training cycle
# alpha is the learning rate 
# X the input samples (each sample is a row of X)
# y is the array of ground truths. For data sample
# x[i] the ground truth is stored in y[i]    
# num_iter is the number of training iterations
def logistic_regression(X, y, alpha, num_iter):
    m, n = X.shape
    # initialize weights with 0
    weights = array.zeros((n,))
    # iterate until max_iter is reached
    for i in range(num_iter):
        # X @ weights denote the inputs to sigmoid
        z = X @ weights
        # calculate sigmoid vectorized for all samples
        # at the same time
        predictions = sigmoid(z)
        # error vector is difference between predictions and ground truth
        error = predictions - y
        # the gradient is X.T@error / m (number of samples) 
        gradient = X.T @ error / m
        # adjust weights using gradient and learning rate alpha
        weights -= alpha @ gradient
    # return the weights calculated in max_iter loops
    return weights
    
# get the weights
weights = logistic_regression(X, y, 0.1, 500)
# get the prediction for a new data sample 
test_X = array([[0.5,0.5]])
test_y = sigmoid(test_X @ weights)
# and print it
print(test_y)




