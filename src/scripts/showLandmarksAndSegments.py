import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import Radiograph

if __name__ == '__main__':
    r = Radiograph.getRadiographs(1)[0]  # type: Radiograph

    r.showWithLandMarks()
    r.showWithSegments()
