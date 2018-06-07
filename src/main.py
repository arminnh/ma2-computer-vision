from Radiograph import Radiograph
from ProcrustesAnalysis import performProcrusteAnaylsis
r = Radiograph(1)
#r.showRadiographWithLandMarks()
performProcrusteAnaylsis(r.landMarks.values())