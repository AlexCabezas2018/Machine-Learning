from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score

class SupportVectorMachine:
    def __init__(self):
        self.model = None
        self.accuracy = 0

    def train(self, x, y, x_val, y_val):
        self.model = self.select_best_model(x, y, x_val, y_val)

    # Esta funcion selecciona los mejores parámetros de C y σ
    def select_best_model(self, x, y, x_val, y_val):
        params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
        best_svm = None
        best_accuracy = 0
        for C in params:
            for sigma in params:
                svm = SVC(kernel='rbf', C=C, gamma=1/(2 * sigma**2))
                svm.fit(x, y.ravel())
                acc = accuracy_score(y_val, svm.predict(x_val))
                if (best_accuracy < acc):
                    best_svm = svm
                    best_accuracy = acc

        return best_svm

    def get_precision(self, x, y):
        return accuracy_score(y, self.model.predict(x))