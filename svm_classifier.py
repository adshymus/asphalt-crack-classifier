from sklearn import svm
from crack_classifier import CrackClassifier
import numpy as np
import matplotlib.pyplot as plt

class SVMClassifier(CrackClassifier):
    def __init__(self, kernel = 'rbf', gamma = 'scale', degree = 3, C = 0.2):
        CrackClassifier.__init__(self)
        self._classifier = svm.SVC(kernel = kernel, degree = degree, gamma= gamma, C = C)

    def Show2DPlot(self, resolution = 0.02):
        super().Show2DPlot()
        # setup marker generator and color map
        data = self._principle_components[:, 0:2]
        self._classifier.fit(data, self._classes)

        # plot the decision surface
        x1_min, x1_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        x2_min, x2_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
        z = self._classifier.predict(np.c_[xx1.ravel(), xx2.ravel()])
        z = z.reshape(xx1.shape)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('SVM decision boundaries')
        plt.contourf(xx1, xx2, z, cmap=self._cmap, alpha=0.4)
        scatter_handle = plt.scatter(data[:, 0], data[:, 1], c=self._classes, cmap=self._cmap, edgecolor='k', s=20)
        plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
        plt.legend(handles=scatter_handle.legend_elements()[0], labels=self._labels)
        plt.show()