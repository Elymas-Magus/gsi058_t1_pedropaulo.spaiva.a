import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import time

def getLocalTime():
    named_tuple = time.localtime() # get struct_time
    return time.strftime("%d/%m/%Y, %H:%M:%S", named_tuple)

class GrayImageResizor:
    def __init__(self, path, dpi, debug = False) -> None:
        self.path = path
        self.dpi = dpi
        self.debug = debug
        self.result = None
        self.image = None
    
    def __logImageSize(self):
        self.__log('Tamanho da imagem', self.image.shape)

    def __log(self, content, *args):
        print('[%s] %s: ' % (getLocalTime(), content), *args)

    def disableDebug(self):
        self.debug = False

    def enableDebug(self):
        self.debug = True

    def setPath(self, path) -> None:
        self.path = path

    def getImage(self):
        return cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)

    def show(self, mode = 'sync') -> None:
        if self.image == None:
            self.image = self.getImage()
            if self.debug:
                self.__logImageSize()
            
        self.showImage(self.image, mode)

    def resize(self, newDpi) -> None:
        self.image = self.getImage()

        if self.dpi < newDpi:
            raise ValueError('O novo DPI deve ser menor que: %s' % self.dpi)

        if self.debug:
            self.__log('Novo DPI', newDpi)
            self.__logImageSize()

        height, width = self.image.shape
        step = self.dpi / newDpi
        roundStep = math.ceil(step)

        newHeight = math.ceil((height / step))
        newWidth = math.ceil((width / step))

        if self.debug:
            self.__log('Tamanho da imagem redimensionada', (newHeight, newWidth))
            
        newImage = np.zeros( (newHeight, newWidth) )

        i = 0
        for pointA in np.arange(0, height, step):
            j = 0
            for pointB in np.arange(0, width, step):
                roundA = math.floor(pointA)
                roundB = math.floor(pointB)
                endPointI = roundA + roundStep
                endPointJ = roundB + roundStep

                block = self.image[roundA:endPointI, roundB:endPointJ]

                newValue = 0
                for line in block:
                    for col in line:
                        newValue += col
                
                newImage[i][j] = np.round(newValue / (roundStep ** 2))
            
                j += 1
            i += 1

        self.result = newImage

    def showImage(self, image, mode = 'sync') -> None:
        ax = plt.subplots()[1]
        ax.imshow(image, cmap = 'gray', vmin = 0, vmax = 255)

        if self.debug:
            self.__log('ploting image ...')
            
        if mode == 'sync':
            plt.show()

    def showResult(self, mode = 'sync') -> None:
        self.showImage(self.result, mode)

    def saveResult(self, filename = 'result', mode = 'sync') -> None:
        if self.debug:
            self.__log('saving image ...')

        plt.savefig(filename, pad_inches = 0, bbox_inches = 'tight')

        if mode == 'sync':
            plt.show()