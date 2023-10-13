import math
from typing import re
import cv2
import tkinter.filedialog as tk
import os
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table

class mahalanobisDistanceClass():

    def __init__(self):
        pass

    def loadFile(self):

        """
        Load File and save in variables
        """

        filetypes_test = (("Image files (*.png, *.jpg, *.bmp)", "*.png *.jpg *.bmp"), ("All Files", "*.*"))
        filenames = tk.askopenfilename(title="Select files...", initialdir=r".\\", filetypes=filetypes_test)

        if filenames:
            name_test = [filenames.split("/")[-1].split(".")[0]]
            filenames = [filenames]
        else:
            name_test = []
            filenames = []

        self.filenames = filenames
        self.name_test = name_test
    def getShapes(self,name):

        """
        Segmentation of every shape inside image and finding properties
        """


        if os.path.exists("dxf_test.txt"):
            os.remove("dxf_test.txt")


        threshold = 100
        self.name = name

        image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]


        #Segmentation of every shape inside image and finding properties


        label_im = label(image)
        regions = regionprops(label_im)


        properties = ["eccentricity", "solidity",'moments_hu','area','perimeter_crofton']
        props = pd.DataFrame(regionprops_table(label_im, image,
                                               properties=properties))


        #I cant get only the first moment of hu so I get all and then remove the ones I dont need
        props = props.drop(['moments_hu-1'], axis=1)
        props = props.drop(['moments_hu-2'], axis=1)
        props = props.drop(['moments_hu-3'], axis=1)
        props = props.drop(['moments_hu-4'], axis=1)
        props = props.drop(['moments_hu-5'], axis=1)
        props = props.drop(['moments_hu-6'], axis=1)

        # calculating compactness
        perimeter = props['perimeter_crofton']
        area = props['area']
        compactness =(4* math.pi *area)/(perimeter*perimeter)
        props['compactness'] = compactness

        # Saving variables that I will need further in the code
        properties = ["eccentricity", "solidity", 'moments_hu-0','compactness']
        self.properties = properties
        self.props = props
        self.regions = regions

        # Ploting shapes found by label function
        # for ii in range(len(regions)):
        #     xcentroid=regions[ii]['centroid'][0]
        #     ycentroid = regions[ii]['centroid'][1]
        #     plt.text(ycentroid, xcentroid, str(ii), fontsize=22,bbox = dict(facecolor = 'red', alpha = 0.5))
        # plt.imshow(label_im,cmap='inferno')
        # plt.show()

    def createModels(self):

        """
        Creating models which are going to be compare with test image
        """

        props = self.props
        lenProps = len(props)
        data = []

        model2 = np.array([
            [0.814348, 0.840176, 0.19556, 0.4851],
            [0.843512, 0.813908, 0.20481, 0.4851],
            [0.833512, 0.823908, 0.22481, 0.5414],
            [0.843512, 0.853908, 0.21481, 0.5402],
            [0.852343, 0.854712, 0.18481, 0.5380],
        ])

        model3 = np.array([
            [0.832950, 0.810834, 0.218203, 0.093253],
            [0.843512, 0.803713, 0.212811, 0.086394],
            [0.823512, 0.797134, 0.211203, 0.074437],
            [0.833512, 0.781141, 0.217811, 0.064982],
            [0.842343, 0.816429, 0.219203, 0.095514],
            [0.827277, 0.812809, 0.218213, 0.05146]
        ])

        model4 = np.array([
            [0.8469, 0.81604, 0.24227, 0.133711],
            [0.8369, 0.81876, 0.24128, 0.130262],
            [0.8430, 0.80636, 0.23863, 0.126989],
            [0.8370, 0.83887, 0.23579, 0.133548],
            [0.8270, 0.79887, 0.25579, 0.120160],

        ])

        model5 = np.array([
            [0.8469, 0.7160, 0.27227, 0.050355],
            [0.8569, 0.6886, 0.28579, 0.044664],
            [0.8189, 0.7484, 0.25531, 0.062366],
            [0.8484, 0.7284, 0.26128, 0.057497],
            [0.8484, 0.6987, 0.28128, 0.044664],

        ])

        model10 = np.array([
            [0.896773, 0.269556, 1.1381, 0.138777],
            [0.866203, 0.278393, 1.1617, 0.128057],
            [0.906599, 0.283453, 1.1031, 0.149432],
            [0.866563, 0.223590, 1.1264, 0.148516],
            [0.856429, 0.275664, 1.1049, 0.138041],
            [0.898827, 0.273271, 1.18163, 0.126382],
        ])
        #Creating data array and model arrays. They are going to be use ahead
        for q in range(lenProps):
            eccentricity = props["eccentricity"][q]
            solidity = props["solidity"][q]
            moments_hu0 = props["moments_hu-0"][q]
            compactness = props['compactness'][q]
            data.append([eccentricity, solidity, moments_hu0, compactness])

        arrayModels = []
        arrayModels.append(model2)
        arrayModels.append(model3)
        arrayModels.append(model4)
        arrayModels.append(model5)
        arrayModels.append(model10)

        self.arraModels = arrayModels
        self.data = data
    def calculateDistance(self):

        """
        Calculating distance and find matching match
        """

        data = self.data
        properties = self.properties
        arrayModels = self.arraModels
        df = []
        dftv = []
        mahalanobisList = []
        minValuesList = []
        threshold = 100


        for i in range(len(arrayModels)):

            model = arrayModels[i]
            buildModel = {}

            #Rearrange variables to fit in panda dataframe
            for j in properties:
                buildModel[j] = []
            for k in range(len(properties)):
                for l in range(len(model)):
                    buildModel[properties[k]].append(model[l][k])

            modelFrame = pd.DataFrame(buildModel, columns=properties)

            if i == 0:
                dftv = pd.DataFrame(data, columns=properties)
                values = pd.DataFrame(data, columns=properties)
            else:
                dftv = dftv

            # calculate mahalanobis values for each shape in image
            mahalanobisList = self.calculateMahalanobis(y=values, data=modelFrame)

            # normalize values and add to dataframe
            #mahalanobisList = self.normalize(mahalanobisList)
            minValuesList.append(min(mahalanobisList))
            dftv[str('Mahalanobis' + str(i))] = mahalanobisList

        #Get min value and matching with model
        minValue=(min(minValuesList))
        minIndex = minValuesList.index(minValue)
        if minValue > threshold:
            minIndex = 'Above '+str(threshold)+' threshold. None found!'

        # print everything to consule
        print(minValue)
        print(minIndex)
        pd.set_option('display.max_columns', None)
        dftv.head()
        print(dftv)

    def calculateMahalanobis(y=None, data=None, cov=None):

        """
        Compute the Mahalanobis Distance between each row of y and the data

        y    : vector or matrix of data with, say, p columns.(DATA WE WANT TO MATCH)
        data : ndarray of the distribution from which Mahalanobis distance of each observation of y is to be computed.(MODEL)
        cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
        """

        mean = data.mean()
        y_mu = y - mean.T
        y_mu_T = y_mu.T
        values_T = data.values.T

        if not cov:
            cov = np.cov(values_T, bias=False,rowvar=True)
        inv_covmat = np.linalg.inv(cov)

        # Formula= multiple everything
        left = np.dot(y_mu, inv_covmat)
        mahal = np.dot(left, y_mu_T)
        return mahal.diagonal()


    def normalize(mahalanobisList):

        """
        Get values in a scale of 0 to 1. DONT! use when values are to far apart! Otherwise most ouputs values will be zero
        """

        maxNumber = max(mahalanobisList, key=int)
        newMahalanobisList = []
        for a in range(len(mahalanobisList)):
            newMahalanobisList.append(mahalanobisList[a] / maxNumber)
        return newMahalanobisList

def main():

    matchShapeClass = mahalanobisDistanceClass()
    matchShapeClass.loadFile()
    fileNames = matchShapeClass.filenames
    for name in fileNames:
        matchShapeClass.getShapes(name)
        matchShapeClass.createModels()
        matchShapeClass.calculateDistance()




if __name__ == '__main__':
    main()