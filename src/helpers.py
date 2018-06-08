from typing import List

from radiograph import Radiograph
def getAllRadiographs():
    radiographs = []
    for i in range(1,31):
        radiographs.append(Radiograph(i))
    return radiographs

def getAllLandmarks(radiographs: List[Radiograph]):
    landmarks = []
    for r in radiographs:
        landmarks += list(r.landMarks.values())
    return landmarks