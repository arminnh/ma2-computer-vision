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
    for k1, v1 in TOOTH_TYPES2.items():
        for k2, v2 in TOOTH_TYPES1.items():
            if v1 == UPPER_TEETH and v2 == CENTRAL_TEETH:
                model = Model(k1 + "-" + k2,
                              landmarks=[l for l in allLandmarks if l.toothNumber in v1 & v2 & LEFT_TEETH])
                model.doProcrustesAnalysis()
                model.buildGrayLevelModels()
                models.append(model)

    # 1.3 Analyze the data using a Principal Component Analysis (PCA), exposing shape class variations
    for model in models:
        model.doPCA()
        # model.reconstruct()

    # Build gray level model for each point of the mean landmarks of the models

    return models


if __name__ == '__main__':
    buildActiveShapeModel()
