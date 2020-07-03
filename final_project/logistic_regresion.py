import numpy as np
from sklearn.metrics import accuracy_score
import scipy.optimize as opt


class LogisticRegresion:
    def __init__(self):
        self.model = None

    def train(self, x, y, reg=0):
        initial_thetas = np.zeros(x.shape[1])
        self.model = opt.fmin_tnc(
            func = log_regresion_regularized_cost,
            x0 = initial_thetas, 
            fprime = log_regresion_regularized_gradient, 
            args = (x, y, reg)
        )[0]

    def fit(self, x, y, x_val, y_val):
        regs = [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
        best_acc = 0
        best_model = None

        for reg in regs:
            initial_thetas = np.zeros(x.shape[1])
            model = opt.fmin_tnc(
                func = log_regresion_regularized_cost,
                x0 = initial_thetas, 
                fprime = log_regresion_regularized_gradient, 
                args = (x, y, reg)
            )[0]

            predictions =  np.round(sigmoid(np.matmul(x_val, np.transpose(model))))
            acc = accuracy_score(y_val, predictions)
            if acc > best_acc:
                best_acc = acc
                best_model = model

        self.model = best_model
            

    def get_precision(self, x, y):
        predictions = np.round(sigmoid(np.matmul(x, np.transpose(self.model))))
        return accuracy_score(y, predictions)


# Funciones auxiliares

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def log_regresion_regularized_cost(thetas, X, Y, Lambda):
    m = np.shape(X)[0]
    sigmoid_X_theta = sigmoid(np.matmul(X, thetas))
    
    term_1_1 = np.matmul(np.transpose(np.log(sigmoid_X_theta)), Y)
    term_1_2 = np.matmul(np.transpose(np.log((1 - sigmoid_X_theta))),(1-Y))
    
    term_1 = - (term_1_1 + term_1_2) / np.shape(X)[0]
    term_2 = Lambda/(2*m) * sum(thetas **2)
    
    return term_1 + term_2

def log_regresion_regularized_gradient(thetas, X, Y, Lambda):
    m = np.shape(X)[0]
    sigmoid_X_theta = sigmoid(np.matmul(X,thetas))
    
    term_1 = np.matmul(np.transpose(X),(sigmoid_X_theta - Y)) /  np.shape(X)[0]
    term_2 = (Lambda/m) * thetas

    return term_1 + term_2

