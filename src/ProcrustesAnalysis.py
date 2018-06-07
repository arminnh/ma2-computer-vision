from Landmark import Landmark
from typing import List
import numpy as np
import random

def performProcrusteAnaylsis(landmarks: List[Landmark]):
    """
    :param landmarks: list of landmarks
    :return:
    """
    ref = random.choice(landmarks)
    for l in landmarks:
        p = l.getPointsWithoutScaleAndTranslation()



