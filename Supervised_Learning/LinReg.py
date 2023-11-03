import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import ssl


ssl._create_default_https_context = ssl._create_unverified_context

'''
Implementation of Linear Regression algorithm with closed form estimation of Beta
'''
class lin_reg: 

    def fit(self, X, Y): 
        col_of_ones = np.ones((X.shape[0], 1))
        self.X_train = X
        X_b = np.hstack((col_of_ones, X))
        self.Y_train = Y
        self.beta_hat = np.linalg.inv(X_b.T @ X_b) @ (X_b.T @ Y)
    
    def predict(self, X): 
        col_of_ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((col_of_ones, X))
        y_pred = X_b @ self.beta_hat
        return y_pred



#test on sample data set boston 

housing = fetch_california_housing() 


X_train, X_test, Y_train, Y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)


sklearnmodel = LinearRegression()
sklearnmodel.fit(X_train,Y_train)
skpred = sklearnmodel.predict(X_test)
model = lin_reg()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

print("The predictions: ")
print(y_pred)
print("The test set: ")
print(Y_test)
print("The MSE: ")
print(mean_squared_error(y_pred, Y_test))
print("The MSE of sklearn model: ")
print(mean_squared_error(skpred, Y_test))


