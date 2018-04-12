import os
import cv2
import numpy as np


def detect_and_save_faces(name, roi_size):
    # define where to look for images and where to save the detected faces
    dir_images = os.path.join("data", name)
    dir_faces = os.path.join("data", name, "faces")
    if not os.path.isdir(dir_faces):
        os.makedirs(dir_faces)

    # put all images in a list
    image_names = [name for name in os.listdir(dir_images) if not name.startswith(".") and name.endswith(".jpg")]
    rectangles = {}

    # detect for each image the face and store this in the faces directory with the same file name
    for image_name in image_names:
        filename = os.path.join(dir_images, image_name)
        new_filename = os.path.join(dir_faces, image_name)
        img = cv2.imread(filename)
        face_cascade = cv2.CascadeClassifier(os.path.join("data", "haarcascade_frontalface_alt.xml"))

        faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=7, minSize=roi_size)
        rectangles[filename] = faces
        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        cv2.imwrite(new_filename, face)

    return rectangles


def show_rectangles_on_images(image_names, rectangles):
    for filename in image_names:
        img = cv2.imread(filename)
        subpath, img_name = os.path.split(filename)

        for (x, y, w, h) in rectangles[filename]:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imwrite(os.path.join(subpath, "rectangles", img_name), img)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)


def load_images_as_rows(image_paths, shape):
    N = len(image_paths)
    P = shape[0] * shape[1]
    X = np.zeros(shape=(N, P))

    for i, face_path in enumerate(image_paths):
        print(i, face_path, cv2.imread(face_path))
        X[i] = cv2.imread(face_path).flatten()
        # read image in grayscale
        # X[i] = cv2.imread(face_path, 0).flatten()

    return X


def do_pca_and_build_model(name, roi_size, images):
    # define where to look for the detected faces
    dir_faces = os.path.join("data", name, "faces")
    faces_paths = [os.path.join(dir_faces, "{}.jpg".format(n)) for n in images]

    # put all faces as data vectors in a N x P data matrix X with N the number of faces and the number of pixels
    X = load_images_as_rows(faces_paths, roi_size)

    # calculate the eigenvectors of X
    mean, eigenvalues, eigenvectors = pca(X, number_of_components=len(faces_paths))

    return [mean, eigenvalues, eigenvectors]


def show_model(model):
    print("TODO Show model", model)


def ttest_images(name, roi_size, images, models):
    # define where to look for the detected faces
    dir_faces = "data/{}/faces".format(name)
    faces_paths = [os.path.join(dir_faces, "{}.jpg".format(n)) for n in images]

    # put all faces as data vectors in a N x P data matrix X with N the number of faces and the number of pixels
    X = load_images_as_rows(faces_paths, roi_size)

    # reconstruct the images in X with each of the models provided and also calculate the MSE
    # store the results as a list of [results_model_reconstructed_X, results_model_MSE]
    results = []
    for model in models:
        projections, reconstructions = project_and_reconstruct(X, model)
        mse = np.mean((X - reconstructions) ** 2, axis=1)
        results.append([reconstructions, mse])

    return results


def pca(X, number_of_components):
    # TODO

    return [mean, eigenvalues, eigenvectors]


def project_and_reconstruct(X, model):
    # TODO

    return [projections, reconstructions]


def show_reconstructed_images(results):
    print("TODO show reconstructed", results)


if __name__ == '__main__':
    ROI_SIZE = (50, 50)  # reasonably quick computation time

    # Detect all faces in all the images of arnold and barack and save them in a subdirectory "faces"
    rectangles_arnold = detect_and_save_faces("arnold", ROI_SIZE)
    rectangles_barack = detect_and_save_faces("barack", ROI_SIZE)

    # visualize detected ROIs overlaid on the original images and copy paste these figures in a document
    show_rectangles_on_images(rectangles_arnold.keys(), rectangles_arnold)  # TODO: comment this line when submitting
    show_rectangles_on_images(rectangles_barack.keys(), rectangles_barack)  # TODO: comment this line when submitting

    # Perform PCA on the previously saved ROIs and build a model for the corresponding person's face
    # making use of a training set. model = [mean, eigenvalues, eigenvectors]
    model_arnold = do_pca_and_build_model("arnold", ROI_SIZE, images=[1, 2, 3, 4, 5, 6])
    model_barack = do_pca_and_build_model("barack", ROI_SIZE, images=[1, 2, 3, 4, 5, 6])

    # visualize these "models" in some way (of your choice) and copy paste these figures in a document
    show_model(model_arnold)  # TODO: comment this line when submitting
    show_model(model_barack)  # TODO: comment this line when submitting

    # Test and reconstruct "unseen" images and check which model best describes it (wrt MSE)
    # results are lists of [results_model_reconstructed_X, results_model_MSE]
    # The correct model-person combination should give best reconstructed images and therefore the lowest MSEs
    results_arnold = ttest_images("arnold", ROI_SIZE, images=[7, 8], models=[model_arnold, model_barack])
    results_barack = ttest_images("barack", ROI_SIZE, images=[7, 8, 9, 10], models=[model_arnold, model_barack])

    # visualize the reconstructed images and copy paste these figures in a document
    show_reconstructed_images(results_arnold)  # TODO: comment this line when submitting
    show_reconstructed_images(results_barack)  # TODO: comment this line when submitting
