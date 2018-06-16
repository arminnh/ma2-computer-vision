import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import Radiograph
import util
from MultiResolutionGUI import MultiResolutionGUI
from Landmark import Landmark
from models.TeethActiveShapeModel import TeethActiveShapeModel


def buildModel(radiographs, resolutionLevels=5, maxLevelIterations=5, grayLevelModelSize=10, sampleAmount=3,
               pClose=0.9, pcaComponents=25):

    mouthLandmarks = []  # list of landmarks that contain all 8 teeth

    for rad in radiographs:
        mouthLandmark = Landmark(np.asarray([]), radiographFilename=None, toothNumber=-1)
        mouthLandmark.radiograph = rad

        for toothNumber, landmark in sorted(rad.landmarks.items(), key=lambda i: i[0]):
            mouthLandmark.points = np.concatenate((mouthLandmark.points, landmark.points))

        mouthLandmarks.append(mouthLandmark)

    model = TeethActiveShapeModel(
        mouthLandmarks=mouthLandmarks,
        resolutionLevels=resolutionLevels,
        maxLevelIterations=maxLevelIterations,
        grayLevelModelSize=grayLevelModelSize,
        sampleAmount=sampleAmount,
        pClose=pClose,
        pcaComponents=pcaComponents
    )

    with util.Timer("> Building multi resolution active shape model: Gray level models"):
        model.buildGrayLevelModels()

    with util.Timer("> Building multi resolution active shape model: Procrustes analysis"):
        model.doProcrustesAnalysis()

    with util.Timer("> Building multi resolution active shape model: PCA"):
        model.doPCA()

    return model


if __name__ == '__main__':
    resolutionLevels = 4
    radiographNumbers = list(range(3))

    with util.Timer("Loading images"):
        radiographs = Radiograph.getRadiographs(numbers=radiographNumbers, resolutionLevels=resolutionLevels)

    with util.Timer("Building multi resolution active shape model"):
        model = buildModel(radiographs, resolutionLevels=resolutionLevels)

    # # Reconstruct some images which were not in the training set to check reconstruction performance
    # for r in Radiograph.getRadiographs([13, 14]):
    #     setLandmark = Landmark(np.asarray([]))
    #
    #     for toothNumber, landmark in sorted(r.landmarks.items(), key=lambda i: i[0]):
    #         setLandmark.points = np.concatenate((setLandmark.points, landmark.points))
    #
    #     model.reconstruct(setLandmark)

    # Load other radiographs for GUI but do not load the ones above again
    with util.Timer("Loading remaining images (without landmarks)"):
        for radiographNumber in range(4):
            if radiographNumber not in radiographNumbers:
                radiographs += Radiograph.getRadiographs([radiographNumber], extra=True)

    gui = MultiResolutionGUI(radiographs, model)
    gui.open()
