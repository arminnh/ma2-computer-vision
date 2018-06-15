import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import Radiograph

if __name__ == '__main__':
    radiographs = Radiograph.getRadiographs(range(15))  # type: Radiograph

    for r in radiographs:
        r.landmarks = {k: v for k, v in r.landmarks.items()}
        r.showWithLandMarks()
        r.plotLandMarksWithGrayLevelModels()
        time.sleep(0.5)
