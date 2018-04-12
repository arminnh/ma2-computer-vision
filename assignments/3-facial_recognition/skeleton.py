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


def save_rectangles_on_images(show, image_names, rectangles):
    for filename in image_names:
        img = cv2.imread(filename)
        subpath, img_name = os.path.split(filename)

        for (x, y, w, h) in rectangles[filename]:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imwrite(os.path.join(subpath, "rectangles", img_name), img)
        if show:
            cv2.imshow("Rectangle on {}".format(img_name), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def load_images_as_rows(image_paths, shape):
    N = len(image_paths)
    P = shape[0] * shape[1]
    X = np.zeros(shape=(N, P), dtype=np.uint8)

    for i, face_path in enumerate(image_paths):
        img = cv2.imread(face_path, 0)
        img_resized = cv2.resize(img, dsize=shape)
        X[i] = img_resized.flatten()

    return X


def pca(X, number_of_components):
    mean = np.mean(X, axis=0).astype(np.uint8)
    eigenvalues = []
    _, eigenvectors = cv2.PCACompute(X, mean=None, maxComponents=number_of_components)

    eigenvectors += abs(eigenvectors.min())
    eigenvectors /= eigenvectors.max()
    eigenvectors *= 255
    eigenvectors = eigenvectors.astype(np.uint8)

    return [mean, eigenvalues, eigenvectors]


def do_pca_and_build_model(name, shape, images):
    # define where to look for the detected faces
    dir_faces = os.path.join("data", name, "faces")
    faces_paths = [os.path.join(dir_faces, "{}.jpg".format(n)) for n in images]

    # put all faces as data vectors in a N x P data matrix X with N the number of faces and the number of pixels
    X = load_images_as_rows(faces_paths, shape)

    # calculate the eigenvectors of X
    mean, eigenvalues, eigenvectors = pca(X, number_of_components=len(faces_paths))

    return [mean, eigenvalues, eigenvectors]


def save_model_images(show, name, model, shape):
    mean, eigenvalues, eigenvectors = model
    dir_model = os.path.join("data", name, "model")

    cv2.imwrite(os.path.join(dir_model, "mean.jpg"), mean.reshape(shape))
    if show:
        cv2.imshow("Mean of {} images".format(name), mean.reshape(shape))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    eigenvector_imgs = [e.reshape(shape) for e in eigenvectors]
    cv2.imwrite(os.path.join(dir_model, "eigenvectors.jpg"), np.concatenate(eigenvector_imgs, axis=1))
    for i, eigenvector_img in enumerate(eigenvector_imgs):
        cv2.imwrite(os.path.join(dir_model, "eigenvector_{}.jpg".format(i)), eigenvector_img)
    if show:
        cv2.imshow("Eigenvectors of {} images".format(name), np.concatenate(eigenvector_imgs, axis=1))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def project_and_reconstruct(X, model):
    # TODO

    return [projections, reconstructions]


def ttest_images(name, shape, images, models):
    # define where to look for the detected faces
    dir_faces = os.path.join("data", name, "faces")
    faces_paths = [os.path.join(dir_faces, "{}.jpg".format(n)) for n in images]

    # put all faces as data vectors in a N x P data matrix X with N the number of faces and the number of pixels
    X = load_images_as_rows(faces_paths, shape)

    # reconstruct the images in X with each of the models provided and also calculate the MSE
    # store the results as a list of [results_model_reconstructed_X, results_model_MSE]
    results = []
    for model in models:
        projections, reconstructions = project_and_reconstruct(X, model)
        mse = np.mean((X - reconstructions) ** 2, axis=1)
        results.append([reconstructions, mse])

    return results


def save_reconstructed_images(show, results):
    print("TODO show reconstructed", results)


if __name__ == '__main__':
    ROI_SIZE = (50, 50)  # reasonably quick computation time

    SHOW_IMAGES = False
    SHOW_IMAGES = True  # TODO: comment this line when submitting

    # Detect all faces in all the images of arnold and barack and save them in a subdirectory "faces"
    # rectangles_arnold = detect_and_save_faces("arnold", ROI_SIZE)
    # rectangles_barack = detect_and_save_faces("barack", ROI_SIZE)

    # visualize detected ROIs overlaid on the original images and copy paste these figures in a document
    # save_rectangles_on_images(SHOW_IMAGES, rectangles_arnold.keys(), rectangles_arnold)
    # save_rectangles_on_images(SHOW_IMAGES, rectangles_barack.keys(), rectangles_barack)

    # Perform PCA on the previously saved ROIs and build a model for the corresponding person's face
    # making use of a training set. model = [mean, eigenvalues, eigenvectors]
    model_arnold = do_pca_and_build_model("arnold", ROI_SIZE, images=[1, 2, 3, 4, 5, 6])
    model_barack = do_pca_and_build_model("barack", ROI_SIZE, images=[1, 2, 3, 4, 5, 6])

    # visualize these "models" in some way (of your choice) and copy paste these figures in a document
    # save_model_images(SHOW_IMAGES, "arnold", model_arnold, ROI_SIZE)
    # save_model_images(SHOW_IMAGES, "barack", model_barack, ROI_SIZE)

    # Test and reconstruct "unseen" images and check which model best describes it (wrt MSE)
    # results are lists of [results_model_reconstructed_X, results_model_MSE]
    # The correct model-person combination should give best reconstructed images and therefore the lowest MSEs
    results_arnold = ttest_images("arnold", ROI_SIZE, images=[7, 8], models=[model_arnold, model_barack])
    # results_barack = ttest_images("barack", ROI_SIZE, images=[7, 8, 9, 10], models=[model_arnold, model_barack])

    # visualize the reconstructed images and copy paste these figures in a document
    save_reconstructed_images(SHOW_IMAGES, results_arnold)
    # save_reconstructed_images(SHOW_IMAGES, results_barack)
