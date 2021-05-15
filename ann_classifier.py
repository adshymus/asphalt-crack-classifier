import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from crack_classifier import CrackClassifier
from sklearn.metrics import confusion_matrix

class ANNClassifier(CrackClassifier):
    def __init__(self):
        super().__init__()
        self._classifier = Sequential()
        self._classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))
        self._classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))
        self._classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    def SplitData(self):
        self._classes = np_utils.to_categorical(self._classes)
        super().SplitData()
    
    def Train(self):
        self._classifier.fit(self._training_set, self._training_set_classes, batch_size = 4, epochs = 50)
    
    def PredictTestingSet(self):
        super().PredictTestingSet()
        self._predictions = (self._predictions == self._predictions.max(axis=1, keepdims=1)).astype(float)
        self._correctly_predicted = np.sum((self._testing_set_classes == self._predictions).all(axis=1))
        self._incorrectly_predicted = self._testing_set_classes.shape[0] - self._correctly_predicted

    def FindConfusionMatrix(self):
        testing_classes = [np.argmax(t) for t in self._testing_set_classes]
        predictions = [np.argmax(p) for p in self._predictions]
        self._confusion_matrix = confusion_matrix(testing_classes, predictions)