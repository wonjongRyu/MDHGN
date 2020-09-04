from LightPipes import *  # Pip install LightPipes

import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class dataGenerator():
    def __init__(self, imageSize, pixelSize, propDistance):
        self.Nx = imageSize
        self.Ny = imageSize
        self.dx = pixelSize*um
        self.Wav = 532*nm
        self.Sizex = self.Nx*self.dx
        self.Sizey = self.Ny*self.dx
        self.z = propDistance*cm  # 전파거리

    def drawDot(self, dotNum, radius):
        Img = np.zeros((512, 512))
        for _ in range(dotNum):
            dotx = np.random.randint(0, 512)
            doty = np.random.randint(0, 512)
            Img = cv2.circle(Img, (dotx, doty), radius, (255, 255, 255), -1)
        return Img

    def drawHologram(self, Img):

        Source = Begin(self.Sizey, self.Wav, self.Ny)

        Source.field = Source.field * Img

        ComplexHologram = Forvard(Source, self.z)

        PhaseHologram = np.angle(ComplexHologram.field)
        AmplitudeHologram = np.abs(ComplexHologram.field)

        RealHologram = np.real(ComplexHologram.field)
        ImaginaryHologram = np.imag(ComplexHologram.field)

        return AmplitudeHologram, PhaseHologram, RealHologram, ImaginaryHologram

    def makeTrainData(self, path, num_of_data):
        count = 0
        for num in range(num_of_data//100):
            for dotnum in range(10, 20, 1):  # 찍을점개수
                for radius in range(0, 10, 1):  # 점크기
                    count += 1
                    Source = self.drawDot(dotnum, radius)
                    [AH, PH, RH, IH] = self.drawHologram(Source)
                    self.saveAsImage(path, Source, RH, IH, count)

    def makeTestData(self, path, num_of_data):
        for idx in range(1, num_of_data+1):
            dotnum = np.random.randint(10, 20)  # 찍을점개수
            radius = np.random.randint(0, 10)  # 점크기
            Source = self.drawDot(dotnum, radius)
            [AH, PH, RH, IH] = self.drawHologram(Source)

            self.saveAsImage(path, Source, RH, IH, idx)

    def saveAsImage(self, path, Source, RH, IH, idx):
        savePath_of_Source = os.path.join(path, 'Source', str(idx)+'.png')
        savePath_of_RH = os.path.join(path, 'RH', str(idx)+'.png')
        savePath_of_IH = os.path.join(path, 'IH', str(idx)+'.png')
        plt.imsave(savePath_of_Source, Source, cmap='gray')
        plt.imsave(savePath_of_RH, RH, cmap='gray')
        plt.imsave(savePath_of_IH, IH, cmap='gray')


def main():
    imageSize = 512
    pixelSize = 3.6  # um
    propDistance = 1  # cm
    num_of_trainData = 1000  # valid = train/10, test= valid/10
    num_of_testData = 20

    dG = dataGenerator(imageSize, pixelSize, propDistance)

    path_of_dataset = "../dataset/" + \
        str(pixelSize)+'um'+str(propDistance)+'cm'

    for phase in ['train', 'valid', 'test']:
        for folder in ['Source', 'RH', 'IH']:
            path = os.path.join(path_of_dataset, phase, folder)
            if not os.path.exists(path):
                os.makedirs(path)

    dG.makeTrainData(path_of_dataset+"/train", num_of_trainData)
    dG.makeTrainData(path_of_dataset+"/valid", num_of_trainData//10)
    dG.makeTestData(path_of_dataset+"/test", num_of_testData)


if __name__ == "__main__":
    main()
