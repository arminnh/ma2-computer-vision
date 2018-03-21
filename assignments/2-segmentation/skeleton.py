""" Cell counting """


import cv2
import numpy as np


def imshow(name, images):
    """ Display images (a list with images all equal number of channels) all together """
    image = np.concatenate(images, axis=1)
    image = cv2.resize(image, dsize=tuple([s // 2 for s in image.shape if s > 3])[::-1])
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def detect_edges(img):
    """ Canny edge detection """
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges_binary = cv2.Canny(img, threshold1=35, threshold2=115)
    edges_on_grayscale_img = grayscale_img.copy()
    edges_on_grayscale_img[edges_binary > 0] = 0
    imshow('Processed Images', [grayscale_img, edges_binary, edges_on_grayscale_img])


def detect_circles(img):
    """ Hough transform to detect circles """
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale_img = cv2.GaussianBlur(grayscale_img, (9, 9), 2)
    circles_on_original_img = img.copy()

    dp = 1
    min_dist = 40
    threshold_lower = 25
    threshold_upper = 85
    min_radius = 30
    max_radius = 120
    circles = cv2.HoughCircles(grayscale_img, cv2.HOUGH_GRADIENT, dp, min_dist, param1=threshold_upper, param2=threshold_lower, minRadius=min_radius, maxRadius=max_radius)

    if circles is not None:
        print(len(circles[0]))
        for x, y, radius in circles[0]:
            cv2.circle(circles_on_original_img, (x, y), radius, (0, 0, 255))

    imshow('All Detected Circles', [circles_on_original_img])
    return circles


def calculate_features(img, circles):
    """ Use the Hough transform to derive a feature vector for each circle """
    ########
    ##TODO##
    ########
    return img_features


def threshold_circles(img, circles, features, thresholds):
    """ Threshold the feature vector to get the "right" circles """
    ########
    ##TODO##
    ########
    imshow('Only Selected Circles', [selected_circles_on_original_image])
    return n


if __name__ == '__main__':

    #read the image
    img = cv2.imread('normal.jpg')

    #show the image
    # imshow('Original Image', [img])

    #do detection
    # detect_edges(img)
    circles = detect_circles(img)
    # features = calculate_features(img, circles)
    # n = threshold_circles(img, circles, features, ((t1, T1), (t2, T2), (t3, T3)))

    #print result
    # print("We counted {} cells.".format(n))
