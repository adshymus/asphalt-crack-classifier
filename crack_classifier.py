import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, classification_report
from crack_preprocessing import CrackDetector
from matplotlib.colors import ListedColormap

class CrackClassifier:
    def __init__(self):
        self._detector = CrackDetector()
        self._scaler = StandardScaler()
        self._pca = PCA(n_components=3)
        self._principle_components = None
        self._classifier = None
        self._confusion_matrix = None
        self._file_data = []
        self._features = []
        self._classes = []
        self._predictions = []
        self._training_set = []
        self._testing_set = []
        self._training_set_classes = []
        self._testing_set_classes = []
        self._correctly_predicted = 0
        self._incorrectly_predicted = 0
        self._labels = ['Alligator/Block', 'Longitudinal/Transverse', 'Non-crack']
        self._cmap = ListedColormap(['red', 'green', 'blue'])

    def LoadData(self, path, sep=";"):
        self._file_data = pd.read_csv(path, sep=sep)
        self._features = self._file_data.iloc[:, 2:].values
        
        self._features = self._scaler.fit_transform(self._features)
        self._classes = self._file_data.iloc[:, 0].values
        self._classes = LabelEncoder().fit_transform(self._classes)
        self._training_set, self._testing_set, self._training_set_classes, self._testing_set_classes = train_test_split(self._features, self._classes, test_size=0.2, random_state=42)
        self._principle_components = self._pca.fit_transform(self._features)

    def ShowExplainedVarianceRatio(self):
        return self._pca.explained_variance_ratio_

    def Train(self):
        self._classifier.fit(self._training_set, self._training_set_classes)

    def Predict(self, image = None):
        if image is None:
            print("Missing argument. Image needed")
            return
        features = self._detector.ExtractFeatures(image)
        arr = []
        arr.append(features)
        features = self._scaler.transform(arr)
        prediction = self._classifier.predict(features)
        return self._labels[prediction[0]]
    
    def PredictFromFolder(self, path):
        if not os.path.isdir(path):
            print("Expected a folder!")
            return
        files = os.listdir(path)

        for file in files:
            try:
                image = mpimg.imread(path + "/" + file)
                prediction = self.Predict(image)
                print(f"image {file} is {prediction}")

            except:
                print(f"error occured when prediction file {file}")

    def ShowTestingResults(self, extra_info = None, save = False):
        self._predictions = self._classifier.predict(self._testing_set)
        self._correctly_predicted = (self._testing_set_classes == self._predictions).sum()
        self._incorrectly_predicted = len(self._predictions) - self._correctly_predicted

        self._confusion_matrix = confusion_matrix(self._testing_set_classes, self._predictions)

        if save:
            file = open("output.txt", "w")
            fun = file.write
        else:
            fun = print

        fun("Test Result for model: {}\n".format(type(self._classifier).__name__))
        fun("_______________________________________________\n")
        if extra_info:
            fun(extra_info + "\n")
        fun('Correctly labelled: {}\n'.format(self._correctly_predicted))
        fun('Incorrectly labelled: {}\n'.format(self._incorrectly_predicted))
        tmp = 100 * self._correctly_predicted / len(self._predictions)
        fun('Accuracy: {:.2f} %\n'.format(tmp))
        string = np.array2string(self._confusion_matrix, precision=2, separator=',')
        fun(string + "\n\n")
        fun('Micro Precision: {:.2f} \n'.format(precision_score(self._testing_set_classes, self._predictions, average='micro')))
        fun('Micro Recall: {:.2f} \n'.format(recall_score(self._testing_set_classes, self._predictions, average='micro')))
        fun('Micro F1-score: {:.2f} \n'.format(f1_score(self._testing_set_classes, self._predictions, average='micro')))

        fun('Macro Precision: {:.2f}\n'.format(precision_score(self._testing_set_classes, self._predictions, average='macro')))
        fun('Macro Recall: {:.2f}\n'.format(recall_score(self._testing_set_classes, self._predictions, average='macro')))
        fun('Macro F1-score: {:.2f}\n'.format(f1_score(self._testing_set_classes, self._predictions, average='macro')))

        fun('Weighted Precision: {:.2f}\n'.format(precision_score(self._testing_set_classes, self._predictions, average='weighted')))
        fun('Weighted Recall: {:.2f}\n'.format(recall_score(self._testing_set_classes, self._predictions, average='weighted')))
        fun('Weighted F1-score: {:.2f}\n'.format(f1_score(self._testing_set_classes, self._predictions, average='weighted')))
        fun(classification_report(self._testing_set_classes, self._predictions, target_names=self._labels))
        disp = ConfusionMatrixDisplay(confusion_matrix=self._confusion_matrix)
        disp.plot()

    def Show2DPlot(self):
        fig, ax = plt.subplots()
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('2 Component PCA')
        scatter_handle = ax.scatter(self._principle_components[:, 0], self._principle_components[:, 1], c=self._classes, cmap=self._cmap, edgecolor='k', s=20)
        ax.grid(color = 'black', linestyle = '--', linewidth = 0.5)
        ax.legend(handles=scatter_handle.legend_elements()[0], labels=self._labels)
        plt.show()
    
    def Show3DPlot(self):
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        plt.title('3 component PCA')
        scatter_handle = ax.scatter3D(self._principle_components[:, 0], self._principle_components[:, 1], self._principle_components[:, 2], c=self._classes, cmap=self._cmap, edgecolor='k', s=40)
        ax.legend(handles=scatter_handle.legend_elements()[0], labels=self._labels)
        plt.show()