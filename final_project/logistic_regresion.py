import numpy as np
from sklearn.metrics import accuracy_score
from scipy import optimize


class LogisticRegresion:
    def __init__(self):
        self.model = None

    def train(self, x, y, reg=0):
        initial_thetas = np.zeros(x.shape[1])
        self.model = optimize.minimize(fun=linearCostGrad,
            x0=initial_thetas,
            args=(x, y, reg),
            method='TNC',
            jac=True)
    
    def get_precision(self, x, y):
        predictions = np.round(self.model.x @ np.transpose(x))
        return accuracy_score(y, predictions)


# Funciones auxiliares
def get_acurracy(Y, Y_pred):
    return np.sum((Y == np.array(Y_pred))) / m

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def cost(thetas, X, Y, reg=0):
    m = X.shape[0]
    H = np.dot(X, thetas)
    cost = (1/(2*m)) * np.sum((H-Y.T)**2) + ( reg / (2 * m) ) * np.sum(thetas[1:]**2)
    return cost

def gradient(thetas, X, Y, reg=0):
    tt = np.copy(thetas)
    tt[0]=0
    m = X.shape[0]
    H = np.dot(X, thetas)
    gradient = ((1 / m) * np.dot(H-Y.T,X)) + ((reg/m) * tt)
    return gradient

def linearCostGrad(thetas,X,Y,reg=0):
    return (cost(thetas,X,Y,reg),gradient(thetas,X,Y).flatten())

