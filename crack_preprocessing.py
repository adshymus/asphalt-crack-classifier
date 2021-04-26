import os
import re
import cv2
import math
import numpy as np
from scipy.optimize import curve_fit
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class CrackDetector:
    def __init__(self):
        self._image = None
        self._gray_image = None
        self._mask = None
        self._blobs = None
        self._numberOfWhitePixels = 0
        self._numberOfBlackPixels = 0
        self._maxElongation = 0
        self._mean = 0
        self._std = 0

    @property
    def Image(self):
        return self._image

    @property
    def ImageGray(self):
        return self._gray_image

    @property
    def StandardDeviation(self):
        return self._std
    
    @property
    def PixelRatio(self):
        return self._numberOfWhitePixels / self._numberOfBlackPixels
    
    @property
    def MaxElongation(self):
        return self._maxElongation
    
    @property
    def Mean(self):
        return self._mean
    
    @property
    def Mask(self):
        return self._mask

    @property
    def Blobs(self):
        return self._blobs
    
    @property
    def NumberOfBlobs(self):
        return len(self._blobs)

    def ShowImage(self):
        plt.imshow(self._image)
    
    def ShowMask(self):
        plt.imshow(self._mask, cmap="gray")

    def LoadImageFromPath(self, path):
        try:
            self._image = mpimg.imread(path)
        except:
            print("Image not found! Check the path")

    def ExtractFeaturesFromFolder(self, path):
        current_path = os.getcwd()
        os.chdir(path)
        names = os.listdir()
        folders = [name for name in names if os.path.isdir(name) and not name.endswith("_processed")]

        for folder in folders:
            os.makedirs(folder + "_processed", exist_ok=True)
            all_files = os.listdir(folder)
            files = [file for file in all_files if not file.endswith('_mask.png')]

            for file in files:
                try:
                    self.LoadImageFromPath(folder + "/" + file)
                    self.ExtractFeatures()
                    mpimg.imsave(folder + "_processed/" + file, self._mask, cmap="gray")
                except:
                    print("folder, file", folder, file)

        os.chdir(current_path)

    def ExtractFeatures(self, image = None):
        if image is not None:
            self._image = image
        if self._image is None:
            print("Load an image with LoadImageFromPath or pass an image to ExtractFeatures")
            return
        self.__segmentImage()
        self.__calcNumberOfPixels()
        self.__calcMaximumElongation()

    def __segmentImage(self):
        # gray scalling, gaussian filter, threshold, median blurring, max pooling then morphology
        self._gray_image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        self._mean = int(np.mean(self._gray_image))
        self._std = np.std(self._gray_image)
        image = cv2.GaussianBlur(self._gray_image, (41, 41), 20)

        lower_range = np.array([0])
        upper_range = np.array([int(self._mean-0.50*self._std)])
        ranged_image = cv2.inRange(image, lower_range, upper_range)
        blurred_image = cv2.medianBlur(ranged_image, 21)
        pooled_image = block_reduce(blurred_image, (10, 10), np.max)

        pooled_image = np.asarray(pooled_image, np.uint8)

        opening_krnl = np.ones((3,3))
        closing_krnl = np.ones((2,2))
        opened_image = cv2.morphologyEx(pooled_image, cv2.MORPH_OPEN, opening_krnl)
        closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, closing_krnl)
        
        contours, _ = cv2.findContours(closed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        blob_mask = np.zeros(closed_image.shape, dtype=np.uint8)
        blobs_mask = np.zeros(closed_image.shape, dtype=np.uint8)

        for i, contour in enumerate(contours):
            c_area = cv2.contourArea(contour)
            if c_area >= 100:
                cv2.drawContours(blob_mask, contours, i, (255, 255, 255), cv2.FILLED)
                blob_mask = cv2.bitwise_and(closed_image, blob_mask)
                blobs_mask = cv2.bitwise_or(blobs_mask, blob_mask)
        
        self._blobs = contours
        self._mask = blobs_mask

    def __calcNumberOfPixels(self):
        numberOfWhitePixels = 0
        numberOfBlackPixels = 0

        for row in self._mask:
            for pixel in row:
                if pixel > 0:
                    numberOfWhitePixels = numberOfWhitePixels + 1
                else:
                    numberOfBlackPixels = numberOfBlackPixels + 1
        
        self._numberOfBlackPixels = numberOfBlackPixels
        self._numberOfWhitePixels = numberOfWhitePixels

    def __calcMaximumElongation(self):
        maxElongation = 0.0
        for index, blob in enumerate(self._blobs):
            rect = cv2.minAreaRect(blob)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            length1 = CrackDetector.__calculateDistance(box[0][0], box[0][1], box[1][0], box[1][1])
            length2 = CrackDetector.__calculateDistance(box[0][0], box[0][1], box[3][0], box[3][1])
            if length1 > length2:
                height = length1
                width = length2
            else:
                height = length2
                width = length1
            if width > 0.0:
                elongation = height / width
                maxElongation = elongation if maxElongation < elongation else maxElongation
        
        self._maxElongation = maxElongation

    def __calculateDistance(x1, y1, x2, y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))