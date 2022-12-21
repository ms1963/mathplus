
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


# X contains the training samples. Each row is an individual example
# y contains the labels for the training data 
X = array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = array([0, 0, 1, 1])

class naive_logistic_regression:
    def __init__(self):
        self.weights = None
        self.sigmoid = Utils.make_vfunc(naive_logistic_regression.sigmoid)

    # the hypothesis is calculated using the sigmoid function
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    # training cycle
    # alpha is the learning rate 
    # X the input samples (each sample is a row of X)
    # y is the array of ground truths. For data sample
    # x[i] the ground truth is stored in y[i]    
    # num_iter is the number of training iterations
    def logistic_regression_fit(self, X, y, alpha, num_iter):
        m, n = X.shape
        # initialize weights with 0
        self.weights = array.zeros((n,))
        # iterate until max_iter is reached
        for i in range(num_iter):
            # X @ weights denote the inputs to sigmoid
            z = X @ self.weights
            # calculate sigmoid vectorized for all samples
            # at the same time
            predictions = self.sigmoid(z)
            # error vector is difference between predictions and ground truth
            error = predictions - y
            # the gradient is X.T@error / m (number of samples) 
            gradient = (X.T @ error).apply(lambda x: x/m)
            # adjust weights using gradient and learning rate alpha
            self.weights -= gradient @ alpha

    
    def predict(self, X, tolerance = 0.5):
        pred = X @ self.weights
        result = []
        for i in range(pred.shape[0]):
            result.append(pred[i] >= tolerance)
        return array(result)
    
# instantiate naive_logistic_regression
lreg = naive_logistic_regression()
# fit the model
lreg.logistic_regression_fit(X, y, 0.1, 1000)
# get the prediction for a new data sample 
test_X = array([[1,1],[1,2],[4,5]])
pred_y = lreg.predict(test_X)
# and print it
print(pred_y)




