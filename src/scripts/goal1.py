import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import active_shape_models
import util
import Radiograph

if __name__ == '__main__':
    radiographNumbers = util.RADIOGRAPH_NUMBERS
    PCAComponents = util.PCA_COMPONENTS
    sampleAmount = util.SAMPLE_AMOUNT

    radiographs = Radiograph.getRadiographs(radiographNumbers)

    active_shape_models.buildActiveShapeModels(radiographs, PCAComponents, sampleAmount)
