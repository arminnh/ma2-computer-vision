import numpy as np

cimport numpy as np
cimport cython

ctypedef np.long DTYPE_t

@cython.boundscheck(False)
def findLineForJawSplit(np.ndarray[long, ndim=2] img, long yMin, long yMax):
    """
    Find the best path to split the jaws in the image starting from position (0, y).
    The path consists of y values, the indices are x values.
    :type img: np.ndarray
    """

    cdef long xMax = img.shape[1]
    yMax = yMax - yMin


    cdef np.ndarray[long] pathX = np.linspace(0, xMax, xMax/20, endpoint=False).astype(long)
    cdef np.ndarray[long] pathY = np.zeros(len(pathX)).astype(long)

    # trellis (y, x, 2) shape. 2 to hold cost and previousY
    cdef np.ndarray[long, ndim=3] trellis = np.full((yMax, len(pathX), 2), np.iinfo(np.int).max)

    # set first column in trellis
    for y in range(yMax):
        trellis[y, 0, 0] = img[y + yMin, 0]
        trellis[y, 0, 1] = y

    cdef long start, end, bestPrevY, bestPrevCost, yWindow
    yWindow = 10

    # forward pass
    for i in range(1, len(pathX)):
        x = pathX[i]
        for y in range(yMax):
            start = y - yWindow if y > yWindow-1 else y
            end = y + yWindow if y < yMax - yWindow else y

            bestPrevY = trellis[start:end+1, i - 1, 0].argmin() + y - yWindow
            bestPrevCost = trellis[bestPrevY, i - 1, 0]

            # new cost = previous best cost + current cost (colour intensity)
            trellis[y, i, 0] = bestPrevCost + img[y + yMin, x]  # + self.transitionCost(bestPrevY, y)
            trellis[y, i, 1] = bestPrevY

    # find the best path, backwards pass
    cdef np.ndarray[long] path = np.zeros((xMax), dtype=long)

    # set first previousY value to set up backwards pass
    cdef long previousY = trellis[:, -1, 0].argmin()

    for i in range(len(pathX) - 1, -1, -1):
        pathY[i] =  previousY + yMin
        previousY = trellis[previousY, i, 1]

    return list(zip(pathX, pathY))
