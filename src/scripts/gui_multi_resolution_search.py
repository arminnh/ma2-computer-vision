import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import Radiograph
import util
from MultiResolutionGUI import MultiResolutionGUI
from Landmark import Landmark
from models.MultiResolutionASM import TeethActiveShapeModel


if __name__ == '__main__':
    resolutionLevels = 5
    radiographNumbers = list(range(0, 20))

    with util.Timer("Loading images"):
        radiographs = Radiograph.getRadiographs(
            numbers=radiographNumbers,
            extra=False,
            resolutionLevels=resolutionLevels,
            withMirrored=True
        )

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
        for radiographNumber in range(25):
            if radiographNumber not in radiographNumbers:
                radiographs += Radiograph.getRadiographs(
                    numbers=[radiographNumber],
                    extra=True,
                    resolutionLevels=resolutionLevels,
                    withMirrored=True,
                )

    gui = MultiResolutionGUI(radiographs, model)
    gui.open()
