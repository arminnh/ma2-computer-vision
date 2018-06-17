import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import models.ToothModel
import models.InitializationModel
import util
import Radiograph
from GUI import GUI

if __name__ == '__main__':
    radiographNumbers = util.RADIOGRAPH_NUMBERS
    PCAComponents = util.PCA_COMPONENTS
    sampleAmount = util.SAMPLE_AMOUNT

    with util.Timer("Loading images"):
        radiographNumbers = list(range(15))
        radiographs = Radiograph.getRadiographs(radiographNumbers)

    initModels = models.InitializationModel.buildModels(radiographs)
    with util.Timer("Building active shape models"):
        models = models.ToothModel.buildModels(radiographs, PCAComponents, sampleAmount)

    # Load other radiographs for GUI but do not load the ones above again
    with util.Timer("Loading remaining images (without landmarks)"):
        for radiographNumber in range(30):
            if radiographNumber not in radiographNumbers:
                radiographs.append(Radiograph.getRadiographs([radiographNumber], extra=True)[0])

    gui = GUI(radiographs, models, initModels)
    gui.open()

    # print(models)
