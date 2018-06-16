import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import Radiograph
import util
from MultiResolutionGUI import MultiResolutionGUI
from Landmark import Landmark
from models.TeethActiveShapeModel import TeethActiveShapeModel


def buildModel(radiographs, resolutionLevels=4, maxLevelIterations=5, grayLevelModelSize=6, sampleAmount=5,
               pClose=0.9, pcaComponents=20):
    # dict of toothnumber -> landmarks for tooth
    individualLandmarks = {}
    # list of landmarks that contain all 8 teeth
    setLandmarks = []

    for rad in radiographs:
        setLandmark = Landmark(np.asarray([]), radiographFilename=None, toothNumber=-1)
        setLandmark.radiograph = rad

        for toothNumber, landmark in sorted(rad.landmarks.items(), key=lambda i: i[0]):
            if toothNumber not in individualLandmarks:
                individualLandmarks[toothNumber] = [landmark]
            else:
                individualLandmarks[toothNumber].append(landmark)

            setLandmark.points = np.concatenate((setLandmark.points, landmark.points))

        setLandmarks.append(setLandmark)

    model = TeethActiveShapeModel(
        individualLandmarks=individualLandmarks,
        setLandmarks=setLandmarks,
        resolutionLevels=resolutionLevels,
        maxLevelIterations=maxLevelIterations,
        grayLevelModelSize=grayLevelModelSize,
        sampleAmount=sampleAmount,
        pClose=pClose,
        pcaComponents=pcaComponents
    )
    model.buildGrayLevelModels()
    model.doProcrustesAnalysis()
    model.doPCA()

    return model


if __name__ == '__main__':
    resolutionLevels = 4
    radiographNumbers = list(range(4))

    with util.Timer("Loading images"):
        radiographs = Radiograph.getRadiographs(numbers=radiographNumbers, resolutionLevels=resolutionLevels)

    with util.Timer("Building multi resolution active shape model"):
        model = buildModel(radiographs, resolutionLevels=resolutionLevels)

    # for r in Radiograph.getRadiographs([14]):
    #     model.reconstruct(r.landmarks)

    # # Load other radiographs for GUI but do not load the ones above again
    # with util.Timer("Loading remaining images (without landmarks)"):
    #     for radiographNumber in range(15):
    #         if radiographNumber not in radiographNumbers:
    #             radiographs.append(Radiograph.getRadiographs([radiographNumber], extra=True)[0])

    gui = MultiResolutionGUI(radiographs, model)
    gui.open()
