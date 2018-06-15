import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import util
import Radiograph
from ToothModel import ToothModel


def buildActiveShapeModels(radiographs, PCAComponents, sampleAmount):
    # 1.1 Load the provided landmarks into your program
    allLandmarks = Radiograph.getAllLandmarksInRadiographs(radiographs)

    # 1.2 Pre-process the landmarks to normalize translation, rotation, and scale differences
    models = []
    for t in util.TEETH:
        models.append(
            ToothModel(
                name=t,
                landmarks=[l for l in allLandmarks if l.toothNumber == t],
                pcaComponents=PCAComponents,
                sampleAmount=sampleAmount,
            )
                .buildGrayLevelModels()
                .doProcrustesAnalysis()
        )

    # 1.3 Analyze the data using a Principal Component Analysis (PCA), exposing shape class variations
    for model in models:
        model.doPCA()
        # model.reconstruct()

    # Build gray level model for each point of the mean landmarks of the models
    return models


if __name__ == '__main__':
    radiographNumbers = util.RADIOGRAPH_NUMBERS
    PCAComponents = util.PCA_COMPONENTS
    sampleAmount = util.SAMPLE_AMOUNT

    radiographs = Radiograph.getRadiographs(radiographNumbers)

    buildActiveShapeModels(radiographs, PCAComponents, sampleAmount)
