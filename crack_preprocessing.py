import os
import re
import cv2
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import curve_fit
from skimage.measure import block_reduce

class CrackDetector:
    def __init__(self):
        self._image = None
        self._gray_image = None
        self._gaussian_image = None
        self._thresholded_image = None
        self._median_image = None
        self._pooled_image = None
        self._opened_image = None
        self._closed_image = None
        self._mask = None
        self._blobs = None
        self._numberOfWhitePixels = 0
        self._numberOfBlackPixels = 0
        self._maxElongation = 0.0
        self._averageElongation = 0.0
        self._maxCompactness = 0.0
        self._averageCompactness = 0.0
        self._maxEccentricity = 0.0
        self._averageEccentricity = 0.0
        self._mean = 0.0
        self._std = 0.0

    @property
    def Image(self):
        return self._image

    @property
    def ImageGray(self):
        return self._gray_image
    
    @property
    def ImageGaussian(self):
        return self._gaussian_image

    @property
    def ImageThreshold(self):
        return self._thresholded_image
    
    @property
    def ImageMedian(self):
        return self._median_image
    
    @property
    def ImagePooling(self):
        return self._pooled_image
    
    @property
    def ImageOpening(self):
        return self._opened_image
    
    @property
    def ImageClosing(self):
        return self._closed_image

    @property
    def StandardDeviation(self):
        return self._std
    
    @property
    def PixelRatio(self):
        if self._numberOfBlackPixels != 0:
            return self._numberOfWhitePixels / self._numberOfBlackPixels
        return 0
    
    @property
    def MaxElongation(self):
        return self._maxElongation
    
    @property
    def AverageElongation(self):
        return self._averageElongation

    @property
    def MaxCompactness(self):
        return self._maxCompactness
    
    @property
    def AverageCompactness(self):
        return self._averageCompactness
    
    @property
    def MaxEccentricity(self):
        return self._maxExcentricity
    
    @property
    def AverageEccentricity(self):
        return self._averageEccentricity
    
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
    
    def ShowImages(self, nrows, save = False, **kwargs):
        remainder = len(kwargs) % nrows

        if remainder > 0:
            print("Make sure the number of rows is divisable by the number of images")
            return

        ncols = len(kwargs) // nrows
        fig, axs = plt.subplots(nrows, ncols, constrained_layout=True)
        keys = kwargs.keys()
        keys = list(reversed(keys))

        if nrows == 1 or ncols == 1:
            num = nrows if nrows > ncols else ncols
            for x in range(0, num):
                key = keys.pop()
                value = kwargs[key]
                key = re.sub(r"(?<=\w)([A-Z]|\d+x\d+)", r" \1", key)
                if key == "Original":
                    axs[x].imshow(value)
                else:
                    axs[x].imshow(value, cmap="gray")
                axs[x].set_title(key)
                axs[x].axis('off')
                axs[x].tight = True
        else:
            for x in range(0, nrows):
                for y in range(0, ncols):
                    key = keys.pop()
                    value = kwargs[key]
                    key = re.sub(r"(?<=\w)([A-Z]|\d+x\d+)", r" \1", key)
                    if key == "Original":
                        axs[x, y].imshow(value)
                    else:
                        axs[x, y].imshow(value, cmap="gray")
                    axs[x, y].set_title(key)
                    axs[x, y].axis('off')
                    axs[x, y].tight = True
        if save:
            fig.savefig('fig.jpg', dpi=150)
    
    def ShowProcessingSteps(self, save = False):
        fig, axs = plt.subplots(3, 3)
        axs[0, 0].imshow(self._image)
        axs[0, 0].set_title("Original")
        axs[0, 1].imshow(self._gray_image, cmap="gray")
        axs[0, 1].set_title("Grayscaling")
        axs[0, 2].imshow(self._gaussian_image, cmap="gray")
        axs[0, 2].set_title("Gaussian filtering")
        axs[1, 0].imshow(self._thresholded_image, cmap="gray")
        axs[1, 0].set_title("Thresholding")
        axs[1, 1].imshow(self._pooled_image, cmap="gray")
        axs[1, 1].set_title("Max pooling")
        axs[1, 2].imshow(self._median_image, cmap="gray")
        axs[1, 2].set_title("Median filtering")
        axs[2, 0].imshow(self._closed_image, cmap="gray")
        axs[2, 0].set_title("Closing")
        axs[2, 1].imshow(self._opened_image, cmap="gray")
        axs[2, 1].set_title("Opening")
        axs[2, 2].imshow(self._mask, cmap="gray")
        axs[2, 2].set_title("Mask")

        for ax in axs.flat:
            ax.axis('off')

        if save:
            fig.savefig('fig.jpg', dpi=150)

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

        data = []

        for folder in folders:
            os.makedirs(folder + "_processed", exist_ok=True)
            all_files = os.listdir(folder)
            files = [file for file in all_files if not file.endswith('_mask.png')]

            for file in files:
                try:
                    image = mpimg.imread(folder + "/" + file)
                    features = self.ExtractFeatures(image)
                    features[:0] = [folder] + [file]
                    data.append(features)
                    # uncomment the following line if you need to save the mask
                    #mpimg.imsave(folder + "_processed/" + file, self.Mask, cmap="gray")
                    
                    for r in range(0, 4):
                        random_brightness = random.randrange(-20, 20)
                        random_contrast = random.uniform(0.9, 1.1)
                        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        gray_image = random_contrast * gray_image + random_brightness
                        gray_image = np.array(gray_image, dtype=np.uint8)
                        features = self.ExtractFeatures(image=gray_image, gray=True)
                        features[:0] = [folder] + [file]
                        data.append(features)
                except Exception as e:
                    print(e)
                    print("folder, file", folder, file)
        
        os.chdir(current_path)
        header = ['Class', 'Mean', 'Standard deviation', 'Max elongation', 'Average elongation', 'Max compactness', 'Average compactess', 'Max eccentricity', 'Average eccentricity', 'Pixel ratio']
        data_frame = pd.DataFrame(data)
        data_frame.to_csv('feature.csv', index=False, sep=";", header=header)

    def ExtractFeatures(self, image = None, gray=False):
        if image is not None:
            self._image = image
        if self._image is None:
            print("Load an image with LoadImageFromPath or pass an image to ExtractFeatures")
            return
        self.__segmentImage(gray=gray)
        self.__calcNumberOfPixels()
        self.__calcMaximumElongation()
        return [self._mean, self._std, self._maxElongation, self._averageElongation, self._maxCompactness, self._averageCompactness, self._maxEccentricity, self._averageEccentricity, self.PixelRatio]

    def __segmentImage(self, gray=False):
        # gray scalling, gaussian filter, threshold, median blurring, max pooling then morphology
        if not gray:
            self._gray_image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        else:
            self._gray_image = self._image

        
        self._mean = np.mean(self._gray_image)
        self._std = np.std(self._gray_image)
        self._gaussian_image = cv2.GaussianBlur(self._gray_image, (101, 101), 0)

        self._thresholded_image = cv2.adaptiveThreshold(self._gaussian_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 189, 7)
        self._thresholded_image = cv2.bitwise_not(self._thresholded_image)

        pooled_image = block_reduce(self._thresholded_image, (10, 10), np.max)
        self._pooled_image = np.asarray(pooled_image, np.uint8)
        self._pooled_image[self._pooled_image < 255] = 0

        self._median_image = self.__AdaptiveMedianFilter(self._pooled_image)
        self._median_image = np.array(self._median_image, dtype=np.uint8)

        self._closed_image = cv2.morphologyEx(self._median_image, cv2.MORPH_CLOSE, np.ones((6,6)))
        self._opened_image = cv2.morphologyEx(self._closed_image, cv2.MORPH_OPEN, np.ones((3,3)))

        contours, _ = cv2.findContours(self._opened_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        blob_mask = np.zeros(self._opened_image.shape, dtype=np.uint8)
        blobs_mask = np.zeros(self._opened_image.shape, dtype=np.uint8)
        blobs = []

        total_compactness = 0.0
        self._maxCompactness = 0.0
        self._averageCompactness = 0.0

        total_eccentricity = 0.0
        self._maxEccentricity = 0.0
        self._averageEccentricity = 0.0

        for i, contour in enumerate(contours):
            c_area = cv2.contourArea(contour)
            if c_area >= 50:
                perimeter = cv2.arcLength(contour, True)
                compactness = (4 * math.pi * c_area) / math.pow(perimeter, 2)
                total_compactness += compactness
                self._maxCompactness = compactness if self._maxCompactness < compactness else self._maxCompactness
                (x,y),(major_axis, minor_axis), angle = cv2.fitEllipse(contour)
                eccentricity = minor_axis / major_axis
                total_eccentricity += eccentricity
                self._maxEccentricity = eccentricity if self._maxEccentricity < eccentricity else self._maxEccentricity
                cv2.drawContours(blob_mask, contours, i, (255, 255, 255), cv2.FILLED)
                blob_mask = cv2.bitwise_and(self._opened_image, blob_mask)
                blobs_mask = cv2.bitwise_or(blobs_mask, blob_mask)
                blobs.append(contour)
        
        if len(contours) != 0:
            self._averageCompactness = total_compactness / len(contours)
            self._averageEccentricity = total_eccentricity / len(contours)
        self._blobs = blobs
        self._mask = blobs_mask

    def __calcNumberOfPixels(self):
        numberOfWhitePixels = 0
        numberOfBlackPixels = 0

        for row in self._mask:
            for pixel in row:
                if pixel > 0:
                    numberOfWhitePixels += 1
                else:
                    numberOfBlackPixels += 1
        
        self._numberOfBlackPixels = numberOfBlackPixels
        self._numberOfWhitePixels = numberOfWhitePixels

    def __calcMaximumElongation(self):
        max_elongation = 0.0
        total_elongation = 0.0
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
                total_elongation = elongation
                max_elongation = elongation if max_elongation < elongation else max_elongation
        
        self._maxElongation = max_elongation

        if len(self._blobs) != 0:
            self._averageElongation = total_elongation / len(self._blobs)

    def __calculateDistance(x1, y1, x2, y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

    def __padding(self, img, pad):
        padded_image = np.zeros((img.shape[0]+2*pad,img.shape[1]+2*pad))
        padded_image[pad:-pad,pad:-pad] = img
        return padded_image

    def __AdaptiveMedianFilter(self, image, s=1, sMax=6):
        H,W = image.shape
        a = sMax//2
        padded_image = self.__padding(image,a)

        filtered_image = np.zeros(padded_image.shape)
        max_range = H+a+1

        for i in range(a, max_range):
            for j in range(a, max_range):
                value = self.__Lvl_A(padded_image,i,j,s,sMax)
                filtered_image[i,j] = value

        return filtered_image[a:-a,a:-a]

    def __Lvl_A(self, mat,x,y,s,sMax):
        window = mat[x-(s//2):x+(s//2)+1,y-(s//2):y+(s//2)+1]
        Zmin = np.min(window)
        Zmed = np.median(window)
        Zmax = np.max(window)

        A1 = Zmed - Zmin
        A2 = Zmed - Zmax

        if A1 > 0 and A2 < 0:
            return self.__Lvl_B(window, Zmin, Zmed, Zmax)
        else:
            s += 2 
            if s <= sMax:
                return self.__Lvl_A(mat,x,y,s,sMax)
            else:
                return Zmed
    
    def __Lvl_B(self, window, Zmin, Zmed, Zmax):
        h,w = window.shape

        Zxy = window[h//2,w//2]
        B1 = Zxy - Zmin
        B2 = Zxy - Zmax

        if B1 > 0 and B2 < 0 :
            return Zxy
        else:
            return Zmed