from LightPipes import *  # Pip install LightPipes

import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def drawDot(N, rad):
    Img = np.zeros((512, 512))
    for _ in range(N):
        dotx = np.random.randint(0, 512)
        doty = np.random.randint(0, 512)
        Img = cv2.circle(Img, (dotx, doty), rad, (255, 255, 255), -1)
    return Img


def drawHologram(Img):
    Nx = 512
    Ny = 512
    Dx = 3.6*um
    Wav = 532*nm
    Sizex = Nx*Dx
    Sizey = Ny*Dx
    PropDis = 5*cm  # 전파거리

    Source = Begin(Sizey, Wav, Ny)

    Source.field = Source.field * Img

    ComplexHologram = Forvard(Source, PropDis)

    PhaseHologram = np.angle(ComplexHologram.field)
    AmplitudeHologram = np.abs(ComplexHologram.field)

    RealHologram = np.real(ComplexHologram.field)
    ImaginaryHologram = np.imag(ComplexHologram.field)

    return AmplitudeHologram, PhaseHologram, RealHologram, ImaginaryHologram


def makeTrainData(sz, path, num_of_data):
    count = 0
    for num in range(num_of_data//100):
        for dotnum in range(10, 20, 1):  # 찍을점개수
            for rad in range(0, 10, 1):  # 점크기
                count += 1
                Source = drawDot(dotnum, rad)
                [AH, PH, RH, IH] = drawHologram(Source)

                savePath_of_Source = os.path.join(
                    path, 'Source', str(count)+'.png')
                savePath_of_RH = os.path.join(path, 'RH', str(count)+'.png')
                savePath_of_IH = os.path.join(path, 'IH', str(count)+'.png')
                plt.imsave(savePath_of_Source, Source, cmap='gray')
                plt.imsave(savePath_of_RH, RH, cmap='gray')
                plt.imsave(savePath_of_IH, IH, cmap='gray')


def makeTestData(sz, path, num_of_data):
    for idx in range(1, num_of_data+1):
        dotnum = np.random.randint(10, 20)  # 찍을점개수
        rad = np.random.randint(0, 10)  # 점크기
        Source = drawDot(dotnum, rad)
        [AH, PH, RH, IH] = drawHologram(Source)

        savePath_of_Source = os.path.join(path, 'Source', str(idx)+'.png')
        savePath_of_RH = os.path.join(path, 'RH', str(idx)+'.png')
        savePath_of_IH = os.path.join(path, 'IH', str(idx)+'.png')
        plt.imsave(savePath_of_Source, Source, cmap='gray')
        plt.imsave(savePath_of_RH, RH, cmap='gray')
        plt.imsave(savePath_of_IH, IH, cmap='gray')


def main():
    sz = 512
    path_of_dataset = "../dataset"
    num_of_trainData = 1000  # valid = train/10, test= valid/10
    num_of_testData = 20

    for phase in ['train', 'valid', 'test']:
        for folder in ['Source', 'RH', 'IH']:
            path = os.path.join(path_of_dataset, phase, folder)
            if not os.path.exists(path):
                os.makedirs(path)

    makeTrainData(sz, path_of_dataset+"/train", num_of_trainData)
    makeTrainData(sz, path_of_dataset+"/valid", num_of_trainData//10)
    makeTestData(sz, path_of_dataset+"/test", num_of_testData)


if __name__ == "__main__":
    main()
