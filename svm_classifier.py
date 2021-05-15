from sklearn import svm
from skimage import measure
from crack_classifier import CrackClassifier
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class SVMClassifier(CrackClassifier):
    def __init__(self, kernel = 'rbf', gamma = 'scale', degree = 3, C = 1.0):
        CrackClassifier.__init__(self)
        self._classifier = svm.SVC(kernel = kernel, degree = degree, gamma= gamma, C = C)

    def ShowTestingResults(self, resolution = 0.02):
        # setup marker generator and color map
        data = self._principle_components[:, 0:2]
        self._classifier.fit(data, self._classes)

        # plot the decision surface
        x1_min, x1_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        x2_min, x2_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        #x3_min, x3_max = self._principle_components[:, 2].min() - 1, self._principle_components[:, 2].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
                            #np.arange(x3_min, x3_max, resolution))
        z = self._classifier.predict(np.c_[xx1.ravel(), xx2.ravel()])
        z = z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, z, cmap=self._cmap, alpha=0.4)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        plt.scatter(data[:, 0], data[:, 1], c=self._classes, cmap=self._cmap)
        plt.show()

    '''
    def ShowTestingResults(self, resolution = 0.2):
        # setup marker generator and color map
        data = self._principle_components[:, 0:3]
        self._classifier.fit(data, self._classes)
        
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=self._classes, cmap=self._cmap)
        z = lambda x,y: (-self._classifier.intercept_[0]-self._classifier.coef_[0][0]*x-self._classifier.coef_[0][1]*y) / self._classifier.coef_[0][2]

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        ### from here i don't know what to do ###
        xx = np.linspace(xlim[0], xlim[1])
        yy = np.linspace(ylim[0], ylim[1])
        zz = np.linspace(zlim[0], zlim[1])
        XX ,YY, ZZ = np.meshgrid(xx, yy, zz)
        xyz = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).T
        Z = self._classifier.predict(xyz)
        Z = Z.reshape(XX.shape)

        # find isosurface with marching cubes
        dx = xx[1] - xx[0]
        dy = yy[1] - yy[0]
        dz = zz[1] - zz[0]
        verts, faces, _, _ = measure.marching_cubes_lewiner(Z, 0, spacing=(1, 1, 1), step_size=2)
        verts *= np.array([dx, dy, dz])
        verts -= np.array([xlim[0], ylim[0], zlim[0]])

        # add as Poly3DCollection
        mesh = Poly3DCollection(verts[faces])
        mesh.set_facecolor('g')
        mesh.set_edgecolor('none')
        mesh.set_alpha(0.3)
        ax.add_collection3d(mesh)
        ax.view_init(20, -45)

        plt.show()'''

s = SVMClassifier('poly')
s.LoadData('features.csv')
s.Train()
s.ShowTestingResults()
