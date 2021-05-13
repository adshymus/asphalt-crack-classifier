from sklearn import svm
from crack_classifier import CrackClassifier

class SVMClassifier(CrackClassifier):
    def __init__(self, kernel = 'rbf', gamma = 'scale', degree = 3, C = 1.0):
        CrackClassifier.__init__(self)
        self._classifier = svm.SVC(kernel = kernel, degree = degree, gamma= gamma, C = C)