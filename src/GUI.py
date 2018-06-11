## Based on code from: https://github.com/Cartucho/OpenLabeling
import math

import util
from preprocess_img import *


class GUI:

    def __init__(self, radiographs, models):
        self.GUI_NAME = 'Computer Vision KU Leuven'
        self.radiographs = radiographs
        self.last_radiograph_index = len(radiographs) - 1
        self.current_radiograph_index = 0
        self.img = None
        self.mouse_x = 0
        self.mouse_y = 0
        self.preprocess = False
        self.models = models

    def open(self):
        self._createWindow()

        TRACKBAR_IMG = self._initTrackBar()

        edges_on = False

        # loop
        while True:
            # clone the img
            tmp_img = self.img.copy()
            height, width = tmp_img.shape[:2]
            if edges_on == True:
                # draw edges
                tmp_img = self.drawEdges(tmp_img)

            cv2.imshow(self.GUI_NAME, tmp_img)
            pressed_key = cv2.waitKey(50)

            """ Key Listeners START """
            if pressed_key == ord('a') or pressed_key == ord('z'):
                # show previous image key listener
                if pressed_key == ord('a'):
                    img_index = self.decreaseIndex(self.current_radiograph_index, self.last_radiograph_index)
                # show next image key listener
                elif pressed_key == ord('z'):
                    img_index = self.increaseIndex(self.current_radiograph_index, self.last_radiograph_index)
                cv2.setTrackbarPos(TRACKBAR_IMG, self.GUI_NAME, img_index)
            # show edges key listener
            elif pressed_key == ord('e'):
                if edges_on == True:
                    edges_on = False
                    cv2.displayOverlay(self.GUI_NAME, "Edges turned OFF!", 1000)

                else:
                    edges_on = True
                    cv2.displayOverlay(self.GUI_NAME, "Edges turned ON!", 1000)
            elif pressed_key == ord("p"):
                self.preprocessCurrentRadiograph()

            # quit key listener
            elif pressed_key == ord('q'):
                break
            """ Key Listeners END """

            if cv2.getWindowProperty(self.GUI_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()

    def _initTrackBar(self):
        # selected image
        TRACKBAR_IMG = 'Image'
        cv2.createTrackbar(TRACKBAR_IMG, self.GUI_NAME, 0, self.last_radiograph_index, self.changeImgIndex)
        # initialize
        self.changeImgIndex(0)
        return TRACKBAR_IMG

    def _createWindow(self):
        # create window
        cv2.namedWindow(self.GUI_NAME, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.GUI_NAME, 1000, 700)
        cv2.setMouseCallback(self.GUI_NAME, self.mouseListener)

    def changeImgIndex(self, x):
        self.current_radiograph_index = x
        # TODO laad de foto in van radiograph
        radiograph = self.radiographs[self.current_radiograph_index]
        self.img = PILtoCV(radiograph.image)
        cv2.displayOverlay(self.GUI_NAME, "Showing image "
                                          "" + str(self.current_radiograph_index) + "/"
                                                                                    "" + str(
            self.last_radiograph_index), 1000)

    def drawEdges(self, tmp_img):
        blur = cv2.bilateralFilter(tmp_img, 3, 75, 75)
        edges = cv2.Canny(blur, 20, 60)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        # Overlap image and edges together
        tmp_img = np.bitwise_or(tmp_img, edges)
        # tmp_img = cv2.addWeighted(tmp_img, 1 - edges_val, edges, edges_val, 0)
        return tmp_img

    def decreaseIndex(self, current_index, last_index):
        self.preprocess = False
        current_index -= 1
        if current_index < 0:
            current_index = last_index
        return current_index

    def increaseIndex(self, current_index, last_index):
        current_index += 1
        self.preprocess = False

        if current_index > last_index:
            current_index = 0
        return current_index

    # mouse callback function
    def mouseListener(self, event, x, y, flags, param):
        global mouse_x, mouse_y

        if event == cv2.EVENT_MOUSEMOVE:
            mouse_x = x
            mouse_y = y
        elif event == cv2.EVENT_LBUTTONDOWN:
            print("click x: {}, y: {}".format(mouse_x, mouse_y))
            # Check contours?
            for m in self.models:
                points = m.translateAndRescaleMean(x, y).getPointsAsTuples()

                for i in range(len(points)):
                    x2, y2 = util.sampleNormalLine(points[i - 1], points[i], points[(i + 1) % len(points)])

                    cv2.line(self.img,
                             (int(x2[0]), int(y2[0])),
                             (int(x2[-1]), int(y2[-1])),
                             (255, 0, 0), 3)

                    origin = (int(points[i][0]), int(points[i][1]))
                    end = (int(points[(i + 1) % len(points)][0]), int(points[(i + 1) % len(points)][1]))

                    cv2.line(self.img, origin, end, (0, 0, 255), 3)

    def preprocessCurrentRadiograph(self):
        radiograph = self.radiographs[self.current_radiograph_index]
        if not self.preprocess:
            self.img = radiograph.preprocessRadiograph([
                PILtoCV,
                bilateralFilter,
                applyCLAHE])
        else:
            self.img = PILtoCV(radiograph.image)

        self.preprocess = not self.preprocess
