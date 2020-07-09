from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score

class SupportVectorMachine:
    def __init__(self):
        self.model = None
        self.accuracy = 0

    def train(self, x, y):
        self.model = SVC(kernel='rbf', gamma='scale')
        self.model.fit(x, y.ravel())

    def fit(self, x, y, x_val, y_val):
        self.model = self.select_best_model(x, y, x_val, y_val)

    # Esta funcion selecciona los mejores parámetros de C y σ
    def select_best_model(self, x, y, x_val, y_val):
        params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
        best_svm = None
        best_accuracy = 0
        for C in params:
            svm = SVC(kernel='rbf', C=C, gamma='scale')
            svm.fit(x, y.ravel())
            acc = accuracy_score(y_val, svm.predict(x_val))
            if (best_accuracy < acc):
                best_svm = svm
                best_accuracy = acc

        return best_svm

    def get_precision(self, x, y):
        return accuracy_score(y, self.model.predict(x))

    def get_fscore(self, x, y):
        prediction = self.model.predict(x)

        number_of_true_positive = sum(y + prediction == 2)
        number_of_predicted_positive = sum(prediction)
        actual_positives = sum(y)

        precision = number_of_true_positive / number_of_predicted_positive
        recall = number_of_true_positive / actual_positives

        return 2 * (precision * recall) / (precision + recall)



