import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

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

    def __init__(self): 
        self.beta_hat = None

    def fit(self, X,Y, L2_penalty): 
        self.X_train = X 
        self.Y_train = Y

        col_of_ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((col_of_ones, X))
        A = np.identity(X_b.shape[1])
        A[0,0] = 0
        self.beta_hat = np.linalg.pinv(X_b.T @ X_b + L2_penalty * A) @ (X_b.T @ Y)
    
    def predict(self, X): 
        col_of_ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((col_of_ones, X))
        y_pred = X_b @ self.beta_hat
        return y_pred

    def get_beta_hat(self):
        return self.beta_hat
    
    def calculate_err(self, y_pred, y_valid): 
        #sums of squares
        return np.sum((y_pred - y_valid)**2)




'''
Implementation of ridge regression, choosing optimal lambda out of a set of lambda
test on the housing data set
'''
'''
Algorithm: Ridge Regression (L2 Regularization) + Cross Validation to pick lambda from a set
1. split data into (X, Y) and (X_test, Y_test)
2. For fold i = 1,2, ..., k 
    1. split (X,Y) into (X_train, Y_train) and (X_valid, Y_valid)
    2. For each candidate lambda in the lambda set: 
        1. Compute beta_lambda on (X_train, Y_train)
        2. compute error (k, lambda) = sum((Y_valid - X_valid beta_lambda)^2
3. Choose lambda_best that minimizes (1/K)*sum_k error(k, lambda)
4. Compute beta_best using lambda_best on (X,Y)
5. Compute beta_best using lambda_best on (X,Y)
'''

def ridge_cv_lambda_selection(X, Y, k_fold, lambdas): 
    fold_size = len(X) // k_fold
    errors = np.zeros(len(lambdas))

    for i in range(k_fold): 
        start = i * fold_size
        end = (i + 1) * fold_size if i != k_fold - 1 else None

        X_valid, Y_valid = X[start:end], Y[start:end]
        X_train = np.concatenate((X[:start], X[end:]), axis = 0)
        Y_train = np.concatenate((Y[:start], Y[end:]), axis = 0)

        for j, L2_penalty in enumerate(lambdas): 
            model = ridge_reg()
            model.fit(X_train,Y_train, L2_penalty)
            y_pred = model.predict(X_valid)
            errors[j]+=model.calculate_err(y_pred,Y_valid)
    errors /= k_fold

    best_lambda_index = np.argmin(errors)
    best_lambda = lambdas[best_lambda_index]
    return best_lambda


'''
1. Split data into (X,Y) and (X test, Y test)
2. For fold k = 1,2, ..., K:
    1. Split (X,Y) into (X train, Y train) and (X valid, Y valid).
    2. Compute betahat1 on (X train, Y train) and
        err1[k] = | Y valid – X valid betahat1 |^2
    3. Compute betahat2 on (X train, log(Y train)) and
        err2[k] = | Y valid – exp( X valid betahat2 ) |^2
3. If sum_k err2[k] < sum_k err1[k], compute final betahat on (X, log(Y)).
    Otherwise,
    compute final betahat on (X, Y).
5. Compute test error on (X test, Y test)

'''

def ridge_cv_log_transformation(X,Y, k_fold, L2_penalty): 
    fold_size = len(X) // k_fold 
    errorsNormal = np.zeros(k_fold)
    errorsLog = np.zeros(k_fold)

    for i in range(k_fold): 
        start = i * fold_size 
        end = (i+1) * fold_size if i!=k_fold-1 else None
        X_valid, Y_valid = X[start:end], Y[start:end]
        X_train = np.concatenate((X[:start], X[end:]), axis = 0)
        Y_train = np.concatenate((Y[:start], Y[end:]), axis = 0) 

        model = ridge_reg()
        model.fit(X_train, Y_train, L2_penalty)
        y_pred_norm = model.predict(X_valid)
        errorsNormal[i]+=model.calculate_err(y_pred_norm, Y_valid)

        model.fit(X_train, np.log(Y_train), L2_penalty)
        y_pred_log = model.predict(X_valid)
        errorsLog[i]+=model.calculate_err(np.exp(y_pred_log), Y_valid)

    sum_norm = np.sum(errorsNormal)
    sum_log = np.sum(errorsLog)

    if sum_log < sum_norm: 
        return "Use Log Transformation"
    return "Do not use log transformation"



data_path = "data/kc_house_data.csv"
data = pd.read_csv(data_path)
print(data.head())


X = data.drop('price', axis=1)
X = np.array(X.drop(['id', 'date'], axis = 1))
Y= np.array(data['price'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state = 42)


print("Original X: ")
print(X.shape)
print("Original Y: ")
print(Y.shape)

#first let us pick an optimal L2_penalty for our model: 

possible_vals = [0.1, 0.2, 0.5, 1, 2, 3, 5, 10, 20, 30, 100]
print("The optimal L2 Penalty we should use: ")
print(ridge_cv_lambda_selection(X_train, Y_train, 5, possible_vals))

L2 = 0.1 
print("Should we use log transformation on our data?")
print(ridge_cv_log_transformation(X_train,Y_train, 5, 0.1))


model = ridge_reg() 
model.fit(X_train, np.log(Y_train), 0.1)
y_pred_log = model.predict(X_test) 
y_pred = np.exp(y_pred_log)
print("The mean squared error after L2 regularization L2 = 0.1 and log transformation on my written from scratch model: ")
print(mean_squared_error(Y_test, y_pred))
print("Now lets compare to sklearn")


sklearn_model = Ridge(alpha = 0.1)
sklearn_model.fit(X_train, np.log(Y_train))
sklearn_y_log_pred = sklearn_model.predict(X_test)
sklearn_y_pred = np.exp(sklearn_y_log_pred)

print("\n\n\n")
print("The mean squared error after L2 regularization L2 = 0.1 and log transformation on sklearn model: ")
print(mean_squared_error(Y_test, sklearn_y_pred))




        









