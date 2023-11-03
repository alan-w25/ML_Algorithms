import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import datasets
from collections import Counter
from sklearn.metrics import mean_squared_error

#Input: 
#n x p Matrix X
#length-n vector Y 
#Output:
#length-n vectors Y pred

def distance(p1, p2): 
    #formula: sqrt(sum(Xij - Xkj)^2)
    #now they are vectorized
    p1,p2 = np.array(p1), np.array(p2)
    return np.sqrt(sum((p1-p2)**2))

'''
Implementation of K-nearest neighbors algorithm for classification and for regression
'''
class KNN: 

    #initializes KNN class
    def __init__ (self, k = 3):
        self.k = k

    #creates the train test splits based on a specifed test size
    def fit(self, X, Y): 
        self.X_train = X 
        self.Y_train = Y

    #predicts classification result
    def predict_classification(self, X): 
        y_pred = [self._predict_classification_one_row(x) for x in X]
        return np.array(y_pred)
    
    #predicts regression result
    def predict_regression(self, X): 
        y_pred = [self._predict_regression_one_row(x) for x in X]
        return np.array(y_pred)
    
    #prediction for one row regression
    def _predict_regression_one_row(self, x_instance): 
        distances = [distance(x_instance, x_train_instance) for x_train_instance in self.X_train]

        k_idx = np.argsort(distances)[:self.k]

        k_neighbor_labels = np.array([self.Y_train[i] for i in k_idx])

        return np.mean(k_neighbor_labels)
        
    #prediction for one row 
    def _predict_classification_one_row(self, x_instance): 
        distances = [distance(x_instance, x_train_instance) for x_train_instance in self.X_train]

        k_idx = np.argsort(distances)[:self.k]

        k_neighbor_labels = [self.Y_train[i] for i in k_idx]

        most_common = Counter(k_neighbor_labels).most_common(1)
        
        return most_common[0][0]



#testing the KNN class on some sample data from iris data set
iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(iris_data, iris_labels, test_size = 0.2, random_state= 42)

model = KNN(k = 10)
model.fit(X_train, Y_train)

print("The X_train shape is:")
print(X_train.shape)
print("The Y_train shape is: ")
print(Y_train.shape)

#training it on the training model
y_pred = model.predict_classification(X_test)

print("The predicted y looks like this: ")
print(y_pred)
print("The test y looks like this: ")
print(Y_test)

print("The MSE: ")
print(mean_squared_error(y_pred, Y_test))