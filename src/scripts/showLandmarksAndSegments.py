import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import util
import Radiograph

if __name__ == '__main__':
    radiographs = Radiograph.getRadiographs(list(range(1, 5)))  # type: Radiograph

    for r in radiographs:
        r.landmarks = {k: v for k, v in r.landmarks.items() if k in util.UPPER_TEETH & util.CENTRAL_TEETH}
        r.plotLandMarksWithGrayLevelModels()
        time.sleep(0.5)
