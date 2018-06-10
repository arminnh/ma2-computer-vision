import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import Radiograph

if __name__ == '__main__':
    radiographs = Radiograph.getRadiographs()  # type: Radiograph

    for r in radiographs:
        r.showWithLandMarks()
        time.sleep(0.5)
