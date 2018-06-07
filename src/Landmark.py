class Landmark:

    def __init__(self, fileName, id):
        self.id = id
        self.points = self._loadPoints(fileName)

    def _loadPoints(self, fileName):
        f = open(fileName, "r")
        p = f.readlines()
        return [(float(p[2*j]),float(p[2*j+1])) for j in range(len(p)/2)]

    def getPoints(self):
        return self.points + [self.points[0]]