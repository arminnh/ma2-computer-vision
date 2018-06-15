import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import util
from GUI import GUI
import Radiograph
import scripts.goal1

if __name__ == '__main__':
    radiographNumbers = util.RADIOGRAPH_NUMBERS
    PCAComponents = util.PCA_COMPONENTS
    sampleAmount = util.SAMPLE_AMOUNT

    radiographNumbers = list(range(3))
    radiographs = Radiograph.getRadiographs(radiographNumbers)

    models = scripts.goal1.buildActiveShapeModels(radiographs, PCAComponents, sampleAmount)

    # Load other radiographs for GUI but do not load the ones above again
    for radiographNumber in range(5):
        if radiographNumber not in radiographNumbers:
            radiographs.append(Radiograph.getRadiographs([radiographNumber], extra=True)[0])

    gui = GUI(radiographs, models)
    gui.open()

    # print(models)
