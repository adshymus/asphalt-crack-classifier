from crack_classifier import CrackClassifier
from keras.models import Sequential
from keras.layers import Dense

class ANNClassifier(CrackClassifier):
    def __init__(self):
        CrackClassifier.__init__(self)
        self._classifier = Sequential()
        self._classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))
        self._classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
        self._classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        self._classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
    
    def Train(self):
        self._classifier.fit(self._training_set, self._training_set_classes, batch_size = 1, epochs = 100)