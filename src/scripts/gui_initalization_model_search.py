import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import util
import Radiograph
from GUI import GUI
from models import ToothModel, InitializationModel

if __name__ == '__main__':
    # radiographNumbers = util.RADIOGRAPH_NUMBERS
    radiographNumbers = list(range(3))
    PCAComponents = util.PCA_COMPONENTS
    sampleAmount = util.SAMPLE_AMOUNT

    with util.Timer("Loading images"):
        radiographs = Radiograph.getRadiographs(radiographNumbers)

    initModels = InitializationModel.buildModels(radiographs, PCAComponents, sampleAmount)
    with util.Timer("Building active shape models"):
        models = ToothModel.buildModels(radiographs, PCAComponents, sampleAmount)

    # Load other radiographs for GUI but do not load the ones above again
    with util.Timer("Loading remaining images (without landmarks)"):
        for radiographNumber in range(4):
            if radiographNumber not in radiographNumbers:
                radiographs.append(Radiograph.getRadiographs([radiographNumber], extra=True)[0])

    gui = GUI(radiographs, models, initModels)
    gui.open()
