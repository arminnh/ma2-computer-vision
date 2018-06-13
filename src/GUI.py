import util
from preprocess_img import *


class GUI:

    def __init__(self, radiographs, models, sampleAmount):
        self.name = "Computer Vision KU Leuven"
        self.radiographs = radiographs
        self.currentRadiograph = None
        self.currentRadiographIndex = 0
        self.models = models
        self.currentModel = None
        self.currentModelIndex = 0
        self.img = None
        self.preprocess = False
        self.currentLandmark = None
        self.showEdges = False
        self.sampleAmount = sampleAmount

    def open(self):
        self.createWindow()
        self.setCurrentImage(self.currentRadiographIndex)
        self.setCurrentModel(self.currentModelIndex)
        self.initTrackBars()

        while True:
            # clone the img
            img = self.img.copy()

            if self.showEdges:
                # draw edges
                img = self.drawEdges(img)

            cv2.imshow(self.name, img)
            y, x = self.img.shape
            cv2.line(self.img, (int(x/2), 0), (int(x/2), int(y)), (255, 255, 255), 3)
            cv2.line(self.img, (0, int(y/2)), (x, int(y/2)), (255, 255, 255), 3)
            for model in self.models:
                landmark = model.landmarks[self.currentRadiographIndex]
                self.drawLandmark(landmark, color=180, thickness=3)

            # Key Listeners
            pressed_key = cv2.waitKey(50)

            # show another image
            if pressed_key == ord("x") or pressed_key == ord("c"):
                if len(self.radiographs) > 1:
                    if pressed_key == ord("x"):
                        self.increaseRadiographIndex(-1)
                    elif pressed_key == ord("c"):
                        self.increaseRadiographIndex(1)
                    cv2.setTrackbarPos("radiograph", self.name, self.currentRadiographIndex)

            # select another model
            if pressed_key == ord("v") or pressed_key == ord("b"):
                if len(self.radiographs) > 1:
                    if pressed_key == ord("v"):
                        self.increaseModelIndex(-1)
                    elif pressed_key == ord("b"):
                        self.increaseModelIndex(1)
                    cv2.setTrackbarPos("model", self.name, self.currentModelIndex)

            # show edges key listener
            elif pressed_key == ord("e"):
                if self.showEdges:
                    self.showEdges = False
                    cv2.displayOverlay(self.name, "Edges turned OFF!", 100)

                else:
                    self.showEdges = True
                    cv2.displayOverlay(self.name, "Edges turned ON!", 100)

            elif pressed_key == ord("p"):
                self.preprocessCurrentRadiograph()

            # quit key listener
            elif pressed_key == ord("q"):
                break

            elif pressed_key == ord("o"):
                self.showOriginalTooth()

            elif pressed_key == ord("n"):
                self.findBetterLandmark()

            if cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()

    def createWindow(self):
        # create window
        cv2.namedWindow(self.name, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.name, 1000, 700)
        cv2.setMouseCallback(self.name, self.mouseListener)
        return self

    def refreshOverlay(self):
        modelName = self.currentModel.name if self.currentModel is not None else self.currentModelIndex

        cv2.displayOverlay(
            self.name,
            "Showing image {}, model {}".format(self.currentRadiographIndex, modelName),
            1000
        )
        return self

    def setCurrentImage(self, idx):
        self.currentRadiographIndex = idx
        self.currentRadiograph = self.radiographs[idx]
        self.img = PILtoCV(self.radiographs[idx].image)
        self.refreshOverlay()
        return self

    def setCurrentModel(self, idx):
        self.currentModelIndex = idx
        self.currentModel = self.models[idx]
        self.refreshOverlay()
        return self

    def initTrackBars(self):
        if len(self.radiographs) > 1:
            cv2.createTrackbar("radiograph", self.name, self.currentRadiographIndex, len(self.radiographs)-1, self.setCurrentImage)

        if len(self.models) > 1:
            cv2.createTrackbar("model", self.name, self.currentModelIndex, len(self.models)-1, self.setCurrentModel)

        return self

    def drawEdges(self, tmp_img):
        blur = cv2.bilateralFilter(tmp_img, 3, 75, 75)
        edges = cv2.Canny(blur, 20, 60)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        # Overlap image and edges together
        tmp_img = np.bitwise_or(tmp_img, edges)
        # tmp_img = cv2.addWeighted(tmp_img, 1 - edges_val, edges, edges_val, 0)
        return tmp_img

    def increaseRadiographIndex(self, amount):
        self.preprocess = False
        self.currentRadiographIndex += amount

        if self.currentRadiographIndex < 0:
            self.currentRadiographIndex = 0

        if self.currentRadiographIndex > len(self.radiographs):
            self.currentRadiographIndex = len(self.radiographs) - 1

        return self

    def increaseModelIndex(self, amount):
        self.currentModelIndex += amount

        if self.currentModelIndex < 0:
            self.currentModelIndex = 0

        if self.currentModelIndex > len(self.models):
            self.currentModelIndex = len(self.models) - 1

        return self

    # mouse callback function
    def mouseListener(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("click x: {}, y: {}".format(x, y))

            self.currentLandmark = self.currentModel.getTranslatedAndInverseScaledMean(x, y)

            self.drawLandMarkWithNormals(self.currentLandmark, grayLevels=False)

    def drawLandmark(self, landmark, color=(0, 0, 255), thickness=1):
        points = landmark.getPointsAsTuples().round().astype(int)

        for i in range(len(points)):
            origin = (points[i][0], points[i][1])
            end = (points[(i + 1) % len(points)][0], points[(i + 1) % len(points)][1])

            cv2.line(self.img, origin, end, color, thickness)

    def drawLandMarkWithNormals(self, landmark, color=(0, 0, 255), grayLevels=True):
        points = landmark.getPointsAsTuples().round().astype(int)

        for i in range(len(points)):
            m = util.getNormalSlope(points[i - 1], points[i], points[(i + 1) % len(points)])
            p = np.asarray(util.sampleNormalLine(m, points[i], self.sampleAmount))
            x2 = p[:, 0]
            y2 = p[:, 1]

            for j in range(len(x2)):
                cv2.line(self.img,
                         (int(x2[j]), int(y2[j])),
                         (int(x2[j]), int(y2[j])),
                         (255, 0, 0), 1)

            origin = (int(points[i][0]), int(points[i][1]))
            end = (int(points[(i + 1) % len(points)][0]), int(points[(i + 1) % len(points)][1]))

            cv2.line(self.img, origin, end, color, 3)

        if grayLevels:
            grayLevelProfiles = landmark.getGrayLevelProfilesForAllNormalPoints(self.sampleAmount, False)
            for r in grayLevelProfiles.values():
                for [pixels, p, p2] in r:
                    print(pixels)
                    for z in range(len(p2)):
                        orig = (int(p2[z][0]), int(p2[z][1]))

                        cv2.line(self.img, orig, orig, int(pixels[z]), thickness=2)

    def preprocessCurrentRadiograph(self):
        if not self.preprocess:
            self.img = self.currentRadiograph.preprocessRadiograph([
                PILtoCV,
                bilateralFilter,
                applyCLAHE
            ])
        else:
            self.img = PILtoCV(self.currentRadiograph.image)

        self.preprocess = not self.preprocess

    def findBetterLandmark(self):
        """ Execute an iteration of "matching model points to target points"
        model points are defined by the model, target points by the 'bestLandmark'
        """
        self.img = PILtoCV(self.currentRadiograph.image)

        self.currentLandmark = self.currentModel.findBetterFittingLandmark(self.currentLandmark, self.currentRadiograph)

        self.currentLandmark = self.currentModel.matchModelPointsToTargetPoints(self.currentLandmark)

        self.drawLandmark(self.currentLandmark, (255, 255, 255))

    def showOriginalTooth(self):
        landmark = self.currentModel.landmarks[self.currentRadiograph]
        self.drawLandMarkWithNormals(landmark)

        grayLevelProfiles, normalizedGrayLevelProfiles, normalPointsOfLandmarkNr \
            = landmark.grayLevelProfileForAllPoints(self.sampleAmount, False)

        self.drawGrayLevelProfiles(grayLevelProfiles, normalPointsOfLandmarkNr)

    def drawGrayLevelProfiles(self, grayLevels, normalPointsOnLandmark):
        for i, pixels in grayLevels.items():
            for j, point in enumerate(normalPointsOnLandmark[i]):
                orig = (int(point[0]), int(point[1]))
                cv2.circle(self.img, orig, 1, int(pixels[j]), thickness=5)
                # cv2.putText(self.img, "{},{}".format(1, 1), orig, cv2.FONT_ITALIC, 0.2, 255)
