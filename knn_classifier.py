from sklearn.neighbors import KNeighborsClassifier
from crack_classifier import CrackClassifier

class KNNClassifier(CrackClassifier):
    def __init__(self, k):
        CrackClassifier.__init__(self)
        self._classifier = KNeighborsClassifier(n_neighbors=k)