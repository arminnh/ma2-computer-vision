def _getBestNextPos(matrix, currPos, r=3):

    (x,y) = currPos

    ix = [ y+i for i in range(r+1) if (y + i) < len(matrix) - 1] + [ y-i for i in range(1,r+1) if y - i> 0]

    costAndPos = []
    for i in ix:
        costAndPos.append((matrix[i][x+1], (x+1,i)))

    return min(costAndPos, key=lambda x:x[0])

def matrixSearch(matrix, startPos):
    """
    Only allowed moves are diagonal or horizontal
    :param matrix:
    :param startPos:
    :return: path, cost
    """
    cost = 0
    # (X,Y)
    currPos = startPos
    path = [startPos]
    while currPos[0] < len(matrix[0])-1:
        (d, nextPos) = _getBestNextPos(matrix, currPos)
        path.append(nextPos)
        cost += d
        currPos = nextPos

    #print(path, cost)
    return path, cost


