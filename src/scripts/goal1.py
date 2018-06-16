import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import models.active_shape_model as asm
import util
import Radiograph

if __name__ == '__main__':
    radiographNumbers = util.RADIOGRAPH_NUMBERS
    PCAComponents = util.PCA_COMPONENTS
    sampleAmount = util.SAMPLE_AMOUNT

    radiographs = Radiograph.getRadiographs(radiographNumbers)

    asm.buildActiveShapeModels(radiographs, PCAComponents, sampleAmount)
