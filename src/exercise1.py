from procrustes_analysis import performProcrustesAnaylsis
from sklearn.decomposition import PCA
from helpers import *

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

