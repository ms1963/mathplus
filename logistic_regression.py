
"""
mathplus.py is licensed under the
GNU General Public License v3.0

Permissions of this strong copyleft license are conditioned on making available complete source code of licensed works and modifications, which include larger works using a licensed work, under the same license. Copyright and license notices must be preserved. Contributors provide an express grant of patent rights.
"""

from mathplus import *

# a second version of logistic regression
# here the error (cross_entropy) is calculated using 
# - (y * log(htheta(x)) + (1-y) * log(1-htheta(x)))
# the prediction for x0,x1,..,xn is 
# htheta(x) = sigmoid(theta0*x0 + theta1*x1 + ... + thetan*xn)
# Or vectorized: h = sigmoid(X @ theta)

class LogisticRegression:
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    def cross_entropy(x,y):
        return -(y*math.log(x,math.e) + (1-y)*math.log(1-x, math.e))
        
    def __init__(self, lr=0.01, max_iter=100000, verbose=False):
        self.lr = lr
        self.max_iter = max_iter
        self.verbose = verbose
        self.sigmoid = Utils.make_vfunc(LogisticRegression.sigmoid)
        self.cross_entropy = Utils.make_vfunc(LogisticRegression.cross_entropy)
    
    def add_bias(self, X):
        intercept = array.ones((X.shape[0], 1))
        return array.concat(intercept, X, axis=0)
        
    def fit(self, X, y):
        X = self.add_bias(X)
        # initialize weights with 0
        self.theta = array.zeros((X.shape[1],))
        for i in range(self.max_iter):
            z = X @ self.theta
            h = self.sigmoid(z)
            gradient = (X.T @ (h - y)).apply(lambda x: x/y.shape[0])
            self.theta = self.theta - gradient * self.lr
            if self.verbose and i % 10000 == 0:
                z = X @ self.theta
                h = self.sigmoid(z)
                print(f'loss: {self.cross_entropy(h,y)}\t')
    
    def predict_prob(self, X):
        return self.sigmoid(X @ self.theta)
            
    def predict(self, X, threshold):
        return self.predict_prob(self.add_bias(X)).apply(lambda x: x  >= threshold)
        
        
lreg = LogisticRegression(lr = 0.01, max_iter = 100000)

# train model with datasets (rows of X) and labels (values of y):
lreg.fit(array([[1,2],[3,4],[5,6],[6,7]]), array([1,1,0,0]))

# make predictions for [1,2] and [5,6]:
print(lreg.predict(array([[1,2],[5,6]]), 0.5))


