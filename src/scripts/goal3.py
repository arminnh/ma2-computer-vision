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

    radiographs = Radiograph.getRadiographs(radiographNumbers)

    models = scripts.goal1.buildActiveShapeModels(radiographs, PCAComponents, sampleAmount)

    gui = GUI(radiographs, models, sampleAmount)
    gui.open()

    # print(models)
