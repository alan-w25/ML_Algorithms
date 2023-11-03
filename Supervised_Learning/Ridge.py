import numpy as np 

'''
Implementation of Ridge Regression Algorithm with cross validation to find optimal lambda value: 
Penalty for really small features
'''

#input: 
# n x p matrix X
# n-length vector Y
# k-length vector lambda values
#implements closed form solution with one lambda value
class ridge_reg: 

    def __init__(self, L2_penalty): 
        self.L2 = L2_penalty

    def fit(self, X,Y): 
        self.X_train = X 
        self.Y_train = Y

        col_of_ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((col_of_ones, X))
        A = np.identity(X.shape[0])
        self.beta_hat = np.inv(X_b.T.dot(X_b) + self.L2.dot(A)).dot(X_b.T).dot(Y)
    
    def predict(self, X): 
        col_of_ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((col_of_ones, X))
        y_pred = X_b @ self.beta_hat
        return y_pred

    def set_l2(self, L2_penalty): 
        self.L2 = L2_penalty

