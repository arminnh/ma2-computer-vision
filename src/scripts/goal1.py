import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.decomposition import PCA
from helpers import *
from procrustes_analysis import performProcrustesAnaylsis


def runExercise():
    allRadiographs = getAllRadiographs()
    # 1.1 load all provided landmarks
    allLandmarks = getAllLandmarks(allRadiographs)

    # 1.2 pre-process landmarks
    newLandmarks = performProcrustesAnaylsis(allLandmarks)

    # 1.3 perform PCA
    pca = PCA(n_components=3)

    data = [l.getPointsAsList() for l in newLandmarks]
    pca.fit(data)


if __name__ == '__main__':
    runExercise()