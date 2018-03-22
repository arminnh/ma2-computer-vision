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
    edges_on_grayscale_img = grayscale_img.copy()

    edges_binary = cv2.Canny(img, threshold1=35, threshold2=115)
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
    circles = cv2.HoughCircles(grayscale_img, cv2.HOUGH_GRADIENT, dp, min_dist, param1=threshold_upper,
                               param2=threshold_lower, minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles = circles[0]

    for x, y, radius in circles:
        cv2.circle(circles_on_original_img, (x, y), radius, (0, 0, 255), 2)

    imshow('All Detected Circles', [circles_on_original_img])
    return circles


def calculate_features(img, circles):
    """ Use the Hough transform to derive a feature vector for each circle """
    rows, cols, channels = img.shape
    img_features = []
    for x, y, radius in circles:
        x, y, radius = int(x), int(y), int(radius)
        x_min, x_max = max(x - radius, 0), min(x + radius + 1, cols)
        y_min, y_max = max(y - radius, 0), min(y + radius + 1, rows)

        region = img[y_min:y_max, x_min:x_max]
        b, g, r = region[:, :, 0].mean(), region[:, :, 1].mean(), region[:, :, 2].mean()
        img_features.append((b, g, r))

        # cv2.circle(img, (x, y), radius, (0, 0, 255))
        # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # imshow('Circles with rectangles for checking', [img])
    return np.array(img_features)


def calculate_thresholds(features):
    """ Calculates thresholds for the "right" circles based on the means and stds of colors """
    b, g, r = features[:, 0], features[:, 1], features[:, 2]

    b_min, b_max = b.mean()-2*b.std(), b.mean()+2*b.std()
    g_min, g_max = g.mean()-2*g.std(), g.mean()+2*g.std()
    r_min, r_max = r.mean()-2*r.std(), r.mean()+2*r.std()
    # print("b_mean: {0:.0f}, b_std: {1:.0f}, b_min:  {2:.0f}, b_max: {3:.0f}".format(b.mean(), b.std(), b_min, b_max))
    # print("g_mean: {0:.0f}, g_std: {1:.0f}, g_min:  {2:.0f}, g_max: {3:.0f}".format(g.mean(), g.std(), g_min, g_max))
    # print("r_mean: {0:.0f}, r_std: {1:.0f}, r_min:  {2:.0f}, r_max: {3:.0f}".format(r.mean(), r.std(), r_min, r_max))

    return b_min, b_max, g_min, g_max, r_min, r_max


def threshold_circles(img, circles, features, thresholds):
    """ Threshold the feature vector to get the "right" circles """
    b_min, b_max, g_min, g_max, r_min, r_max = thresholds
    selected_circles_on_original_image = img.copy()
    n = 0

    for (x, y, radius), (b, g, r) in zip(circles, features):
        cv2.circle(selected_circles_on_original_image, (x, y), radius, (0, 0, 255), 2)

        if b_min <= b <= b_max and g_min <= g <= g_max and r_min <= r <= r_max:
            n += 1
            cv2.circle(selected_circles_on_original_image, (x, y), radius, (0, 255, 0), 2)

    imshow('Only Selected Circles', [selected_circles_on_original_image])
    return n


if __name__ == '__main__':
    # read the image
    img = cv2.imread('normal.jpg')
    print("Image shape:", img.shape)

    # show the image
    imshow('Original Image', [img])

    # do detection
    detect_edges(img)
    circles = detect_circles(img)
    print("Circles detected: ", len(circles))
    features = calculate_features(img, circles)
    n = threshold_circles(img, circles, features, calculate_thresholds(features))

    # print result
    print("We counted {} cells.".format(n))
