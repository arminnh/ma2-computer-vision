import cv2, os
import numpy as np

    
def detect_and_save_faces(name, roi_size):
    
    # define where to look for images and where to save the detected faces
    dir_images = "data/{}".format(name)
    dir_faces = "data/{}/faces".format(name)
    if not os.path.isdir(dir_faces): os.makedirs(dir_faces)  
    
    # put all images in a list
    names_images = [name for name in os.listdir(dir_images) if not name.startswith(".") and name.endswith(".jpg")] # can vary a little bit depending on your operating system
    
    # detect for each image the face and store this in the face directory with the same file name as the original image
    ## TODO ##
            
    
def do_pca_and_build_model(name, roi_size, numbers):
    
    # define where to look for the detected faces
    dir_faces = "data/{}/faces".format(name)
    
    # put all faces in a list
    names_faces = ["0_{}.jpg".format(n) for n in numbers]
    
    # put all faces as data vectors in a N x P data matrix X with N the number of faces and P=roi_size[0]*roi_size[1] the number of pixels
    ## TODO ##
        
    # calculate the eigenvectors of X
    mean, eigenvalues, eigenvectors = pca(X, number_of_components=N)
    
    return [mean, eigenvalues, eigenvectors]
    

def test_images(name, roi_size, numbers, models):

    # define where to look for the detected faces
    dir_faces = "data/{}/faces".format(name)
    
    # put all faces in a list
    names_faces = ["0_{}.jpg".format(n) for n in numbers]
    
    # put all faces as data vectors in a N x P data matrix X with N the number of faces and P=roi_size[0]*roi_size[1] the number of pixels
    ## TODO ##
        
    # reconstruct the images in X with each of the models provided and also calculate the MSE
    # store the results as [[results_model_arnold_reconstructed_X, results_model_arnold_MSE], [results_model_barack_reconstructed_X, results_model_barack_MSE]]
    results = []
    for model in models:
        projections, reconstructions = project_and_reconstruct(X, model)
        mse = np.mean((X - reconstructions) ** 2, axis=1)
        results.append([reconstructions, mse])

    return results
    

def pca(X, number_of_components):
    
    ## TODO ##
    
    return [mean, eigenvalues, eigenvectors]


def project_and_reconstruct(X, model):
    
    ## TODO ##
    
    return [projections, reconstructions]


if __name__ == '__main__':
    
    roi_size = (50, 50) # reasonably quick computation time
    
    # Detect all faces in all the images in the folder of a person (in this case "arnold" and "barack") and save them in a subfolder "faces" accordingly
    detect_and_save_faces("arnold", roi_size=roi_size)
    detect_and_save_faces("barack", roi_size=roi_size)
    
    # visualize detected ROIs overlayed on the original images and copy paste these figures in a document 
    ## TODO ## # please comment this line when submitting
    
    # Perform PCA on the previously saved ROIs and build a model=[mean, eigenvalues, eigenvectors] for the corresponding person's face making use of a training set
    model_arnold = do_pca_and_build_model("arnold", roi_size=roi_size, numbers=[1, 2, 3, 4, 5, 6])
    model_barack = do_pca_and_build_model("barack", roi_size=roi_size, numbers=[1, 2, 3, 4, 5, 6])
    
    # visualize these "models" in some way (of your choice) and copy paste these figures in a document
    ## TODO ## # please comment this line when submitting
    
    # Test and reconstruct "unseen" images and check which model best describes it (wrt MSE)
    # results=[[results_model_arnold_reconstructed_X, results_model_arnold_MSE], [results_model_barack_reconstructed_X, results_model_barack_MSE]]
    # The correct model-person combination should give best reconstructed images and therefor the lowest MSEs
    results_arnold = test_images("arnold", roi_size=roi_size, numbers=[7, 8], models=[model_arnold, model_barack])
    results_barack = test_images("barack", roi_size=roi_size, numbers=[7, 8, 9, 10], models=[model_arnold, model_barack])
    
    # visualize the reconstructed images and copy paste these figures in a document
    ## TODO ## # please comment this line when submitting