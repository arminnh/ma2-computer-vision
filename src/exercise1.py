from typing import List

from radiograph import Radiograph
from procrustes_analysis import performProcrusteAnaylsis
from sklearn.decomposition import PCA


def loadAllRadiographs():
    radiographs = []
    for i in range(1,31):
        radiographs.append(Radiograph(i))
    return radiographs

def getAllLandmarks(radiographs: List[Radiograph]):
    landmarks = []
    for r in radiographs:
        landmarks += list(r.landMarks.values())
    return landmarks

def runExercise():
    allRadioGraphs = loadAllRadiographs()
    # 1.1 load all provided landmarks
    allLandmarks = getAllLandmarks(allRadioGraphs)

    # 1.2 pre-process landmarks
    newLandmarks = performProcrusteAnaylsis(allLandmarks)

    # 1.3 perform PCA
    pca = PCA(n_components=3)

    data = [l.getPointsAsList() for l in newLandmarks]
    pca.fit(data)


