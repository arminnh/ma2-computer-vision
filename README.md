# ma2-computer-vision

As stated in our report we have implemented two different ways to initialize the active shape models.
Therefore we have created two different GUI's such that the user can test both of these initialization methods.
Both scripts to run these GUI's can be found in src/scripts under the name 'gui_initalization_model_search' and
'gui_multi_resolution_search'.
To run one of these two scripts just run the following command:

    python3 script_name.py

# User guide for 'gui_initalization_model_search'
When opening the GUI the user can see three things:
1. The current image.
2. A slider to select the image the user wants
3. A slider to select the tooth model the user wants to fit. 0 indicating the first tooth and 7 indicating the last tooth.

If a user wants to try to fit a model himself/herself all he/she needs to do is click on the image at the desired location.
This will draw an initial landmark at that position.
To iteratively improve this landmark press on the key 'n' untill the desired result.
If the placement of the landmark was wrong just click on an other location and restart the process.

If the user is interested to see the initial placement model at work (model 1 in the paper) the user can press on the key 'i'.
This will automatically place the placement model at the correct location.
To improve on this model the user can press 'n' untill the desired result.
Then the user can use this converged model as initial placement by clicking on their desired position in this model.
This will draw the model for a separate tooth at that location. The user can then click on the key 'n' to converge this model.

If the user wants to automatically segment a tooth press on the key 'a' and wait until the process is done.

# User guide for 'gui_multi_resolution_search'