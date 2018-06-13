import os

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from util import *
import Radiograph
from Model import Model


def buildActiveShapeModel(radiographs=None):
    if radiographs is None:
        radiographs = list(range(1, 15))

    allRadiographs = Radiograph.getRadiographs(radiographs)
    # 1.1 Load the provided landmarks into your program
    allLandmarks = Radiograph.getAllLandmarksInRadiographs(allRadiographs)

    # 1.2 Pre-process the landmarks to normalize translation, rotation, and scale differences
    models = []
    model = Model("test", landmarks=[l for l in allLandmarks if l.toothNumber in [6]])
    model.buildGrayLevelModels()
    model.doProcrustesAnalysis()
    models.append(model)

    # 1.3 Analyze the data using a Principal Component Analysis (PCA), exposing shape class variations
    for model in models:
        model.doPCA()
        #model.reconstruct()

    # Build gray level model for each point of the mean landmarks of the models

    return models


if __name__ == '__main__':
    buildActiveShapeModel()
