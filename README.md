# Rubiks-Cube-Inference-Model


## Modify color_dataset_modifier.py to add data for a new type of Rubik's cube . Make sure to create 6 files for each color called "blue_rgb", "red_rgb", etc. Call the function getData for each of the 6 different colors on the cube. Position the solved face of the current color in the box on the webcam and press "d" once done. 

Preferably, collect multiple samples for each color at different times of the day for maximum accuracy. Note: If there are any more or less than 3 samples for each color, some numbers might need to be adjusted for the program not to produce an error. 

## Run the mlmodel.py file to train the Random Forest Classifier model on the data. Note: This might take 1 - 2 minutes. 

## Finally, run the recognition.py file and the model should accurately predict the colors of each face!
