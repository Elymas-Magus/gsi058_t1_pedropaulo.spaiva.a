import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

class Resizor:
    def __init__(self, path) -> None:
        self.path = path
        self.result = None

    def setPath(self, path):
        self.path = path

    def showImage(img):
        plt.imshow(img, cmap = 'gray', vmin = 0, vmax = 255)

    def getImage(self):
        return cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)

    def resize(self, oldDpi, newDpi, log = False):
        self.img = self.getImage()

        if log:
            print('Tamanho da imagem: ', self.img.shape)

        height, width = self.img.shape
        step = oldDpi / newDpi
        roundStep = math.ceil(step)

        newHeight = math.ceil((height / step))
        newWidth = math.ceil((width / step))

        if log:
            print('Tamanho da imagem redimensionada: ', (newHeight, newWidth))
            
        newImage = np.zeros((newHeight, newWidth))

        i = 0
        for pointA in np.arange(0, height, step):
            j = 0
            for pointB in np.arange(0, width, step):
                roundA = math.floor(pointA)
                roundB = math.floor(pointB)
                endPointI = roundA + roundStep
                endPointJ = roundB + roundStep

                block = self.img[roundA:endPointI, roundB:endPointJ]

                newValue = 0
                for line in block:
                    print(len(line))
                    for col in line:
                        newValue += col
                
                # print(i, j)
                newImage[i][j] = newValue / (roundStep ** 2)

                j += 1
            i += 1

        self.result = newImage

    def showResult(self):
        imgRGB = cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB)
        # plt.imshow(imgRGB)
        plt.imshow(imgRGB)