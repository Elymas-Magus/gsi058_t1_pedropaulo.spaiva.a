import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from os import path

def getLocalTime():
    named_tuple = time.localtime() # get struct_time
    return time.strftime("%d/%m/%Y, %H:%M:%S", named_tuple)

class GrayImageManipulator:
    def __init__(self, path, quantization = 8, debug = False) -> None:
        self.path = path
        self.quantization = quantization
        self.debug = debug
        self.result = None
        self.image = None

    def log(self, content, *args):
        print('[%s] %s: ' % (getLocalTime(), content), *args)
    
    def logImageSize(self):
        self.log('Tamanho da imagem', self.image.shape)

    def disableDebug(self) -> None:
        self.debug = False

    def enableDebug(self) -> None:
        self.debug = True

    def setPath(self, path) -> None:
        self.path = path

    def getPath(self, path):
        return path

    def setImage(self, image) -> None:
        self.image = image

    def getImage(self):
        return self.image

    def getResult(self):
        return self.result

    def configImage(self):
        image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)

        if not path.isfile(self.path):
            raise TypeError('O caminho não corresponde a um arquivo')

        if image is None:
            raise TypeError('O caminho não corresponde a uma imagem')
        
        self.image = image

    def show(self, mode = 'sync') -> None:
        if self.image == None:
            self.configImage()
            
            if self.debug:
                self.logImageSize()
            
        self.showImage(self.image, mode = mode)

    def showImage(self, image, cmap = 'gray', mode = 'sync') -> None:
        ax = plt.subplots()[1]
        ax.imshow(image, cmap = cmap, vmin = 0, vmax = 255)

        if self.debug:
            self.log('ploting image ...')
            
        if mode == 'sync':
            plt.show()

    def showResult(self, cmap = 'gray', mode = 'sync') -> None:
        self.validateResult()
        self.showImage(self.result, cmap = cmap, mode = mode)

    def saveResult(self, filename = 'result', cmap = 'gray', mode = 'sync') -> None:
        self.validateResult()
        
        ax = plt.subplots()[1]
        ax.imshow(self.result, cmap = cmap, vmin = 0, vmax = 255)

        if self.debug:
            self.log('saving image ...')

        plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0)

        if mode == 'sync':
            plt.show()

    def validateResult(self):
        if self.result is None:
            raise ValueError('O resultado não pode ser nulo')