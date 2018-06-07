class Landmark:

    def __init__(self, fileName, id):
        self.id = id
        self.points = self._loadPoints(fileName)

    def _loadPoints(self, fileName):
        f = open(fileName, "r")
        p = f.readlines()
        return [float(x) for x in p]
