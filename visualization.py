import enum
import pickle
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA 
from sklearn.metrics import ConfusionMatrixDisplay

#plt.style.use('seaborn')


def v1():
    folders = os.listdir('data')

    heights = []
    widths = []

    h = 0
    w = 0
    cnt = 0

    for folder in folders:
        paths = os.listdir(os.path.join('data', folder))
        for path in paths:
            im = cv2.imread(os.path.join('data', folder, path))
            heights.append(im.shape[0])
            widths.append(im.shape[1])
            h += im.shape[0]
            w += im.shape[1]
            cnt += 1


    plt.hist(heights, bins=range(100, 360, 20), alpha=0.5, label='heights')
    plt.hist(widths, bins=range(100, 360, 20), alpha=0.5, label='widths')
    plt.legend()
    plt.show()

    print(h / cnt)
    print(w / cnt)



def v2():
    sizes = [1, 4, 8, 16, 32]
    times1 = [0.05730, 0.02671, 0.02579, 0.02731, 0.02891]
    times2 = [0.04536, 0.01180, 0.01102, 0.01491, 0.02260]
    baseline = 0.61532

    speedup1 = [baseline / t for t in times1]
    speedup2 = [baseline / t for t in times2]

    print(speedup1, speedup2)

    plt.plot(sizes, speedup1, label='naive')
    plt.plot(sizes, speedup2, label='optimized')
    plt.legend()
    plt.show()


def v3():
    sizes = [16, 64, 256, 1024]
    times = [0.8032, 0.4028, 0.4047, 0.4052]
    baseline = 17.27

    speedup = [baseline / t for t in times]

    print(speedup)
    
    plt.plot(sizes, speedup)
    plt.show()


def v4():
    arr = []
    with open('BowSceneRecognition/centers.txt', 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            arr.append([float(x) for x in line.strip().split(' ')])
    
    arr = np.array(arr)

    pca = PCA(3)
    pca.fit(arr)

    arr2d = pca.transform(arr)

    ax = plt.axes(projection='3d')
    ax.scatter(arr2d[:, 0], arr2d[:, 1], arr2d[:, 2])
    plt.show()


def v5():
    sizes = [3, 5, 10, 15, 25, 50]
    times = [0.530547, 0.543408, 0.588424, 0.55627, 0.533762, 0.498392]
    
    plt.plot(sizes, times)
    plt.show()


def v6():
    conf = np.zeros([8, 8])
    
    with open('conf.txt', 'r') as f:
        for i, line in enumerate(f):
            for j, k in enumerate(line.strip().split(' ')):
                conf[i][j] = int(k)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf, 
        display_labels=['airport', 'auditorium', 'bedroom', 'campus', 'desert', 'football_stadium', 'landscape', 'rainforest'])
    disp.plot()
    plt.show()


v6()
