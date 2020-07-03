import numpy as np
import scipy.optimize as opt
from sklearn.metrics import accuracy_score

MAX_ITERATIONS = 80

class NeuralNetWork:

    def __init__(self):
        self.model = None

    def train(self, x, y):
        random_theta1 = random_weights(9, 32)
        random_theta2 = random_weights(33, 1)
        params_rn =  np.concatenate((np.ravel(random_theta1), np.ravel(random_theta2)))

        self.model = opt.minimize(
                fun = backprop,
                x0 = params_rn, 
                args = (8, 32, 1, x, y, 1), 
                method = 'TNC', 
                jac = True, 
                options = {'maxiter': MAX_ITERATIONS}
            )


    def fit(self, x, y, x_val, y_val):
        self.model = self.select_best_model_based_on_lambda(x, y, x_val, y_val)


    def select_best_model_based_on_lambda(self, x, y, x_val, y_val):
        lambdas = np.linspace(0, 1, 10)
        random_theta1 = random_weights(9, 32)
        random_theta2 = random_weights(33, 1)
        params_rn =  np.concatenate((np.ravel(random_theta1), np.ravel(random_theta2)))

        best_model = None
        best_acc = 0
        for lbda in lambdas:

            fmin = opt.minimize(
                fun = backprop,
                x0 = params_rn, 
                args = (8, 32, 1, x, y, lbda), 
                method = 'TNC', 
                jac = True, 
                options = {'maxiter': MAX_ITERATIONS}
            )

            theta1_opt = np.reshape(fmin.x[:32 * (8 + 1)], (32, (8 + 1)))
            theta2_opt = np.reshape(fmin.x[32 * (8 + 1):], (1, (32 + 1)))

            predictions = forward_propagate(x_val, theta1_opt, theta2_opt)
            acc = accuracy_score(y_val, np.round(predictions[4]))

            if acc > best_acc:
                best_acc = acc
                best_model = fmin
            
        return best_model

    def get_precision(self, x, y):
        theta1_opt = np.reshape(self.model.x[:32 * (8 + 1)], (32, (8 + 1)))
        theta2_opt = np.reshape(self.model.x[32 * (8 + 1):], (1, (32 + 1)))

        predictions = forward_propagate(x, theta1_opt, theta2_opt)
        return accuracy_score(y, np.round(predictions[4]))

# Funciones auxiliares

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def random_weights(L_in, L_out, epsilon = 0.0001):
    np.random.seed(0)
    return np.random.random((L_in, L_out)) * (2 * epsilon) - epsilon

# Forward propagation
def forward_propagate(x, theta1, theta2):
    m = x.shape[0]

    a1 = np.hstack([np.ones([m, 1]), x])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)

    return a1, z2, a2, z3, h

# Gradient function
def gradient(x, y, theta1,theta2, reg = 0):
    m = x.shape[0]
    
    delta1 = np.zeros(theta1.shape)  # (32, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    a1, z2, a2, z3, h = forward_propagate(x, theta1, theta2)
    
    for t in range(m):    
        a1t = a1[t, :] 
        a2t = a2[t, :] 
        ht = h[t, :]  
        yt = y[t]  
        d3t = ht - yt  
        d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t)) 
        delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        delta1[1:, :] += (theta1[1:, :] * (reg / m))
        delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])
        delta2[1:, :] += (theta2[1:, :] * (reg / m))
    
    return np.concatenate((np.ravel(delta1 / x.shape[0]) , np.ravel(delta2 / x.shape[0])))

def cost(x, y, theta1, theta2, num_etiquetas, reg=0):
    a1, z2, a2, z3, h = forward_propagate(x, theta1, theta2)
    m = x.shape[0]

    J = 0
    for i in range(m):
        J += np.sum(-y[i] * np.log(h[i]) - (1 - y[i]) * np.log(1 - h[i]))
    J = J / m

    sum_theta1 = np.sum(np.square(theta1[:, 1:]))
    sum_theta2 = np.sum(np.square(theta2[:, 1:]))

    term_3 = (sum_theta1 + sum_theta2) * (reg / (2 * m))

    return J + term_3


# Cost function
def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, x, y, reg):
    """ backprop devuelve el coste y el gradiente de una red neuronal de dos capas """
    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))

    return cost(x, y, theta1, theta2, num_etiquetas, reg), gradient(x, y, theta1, theta2, reg)