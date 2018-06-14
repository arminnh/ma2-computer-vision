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

    radiographNumbers = list(range(8))
    radiographs = Radiograph.getRadiographs(radiographNumbers)

    models = scripts.goal1.buildActiveShapeModels(radiographs, PCAComponents, sampleAmount)

    radiographs = Radiograph.getRadiographs(list(range(30)), extra=True)
    gui = GUI(radiographs, models)
    gui.open()

    # print(models)
