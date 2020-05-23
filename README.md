
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The libraries required for the successful execution of this code are mentioned in requirements.txt. In order to install all the libraries:
`pip install -r requirements.txt`

Download the dog images from [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) and extract into `Data/dogImages` folder.

Download the human images from [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip) and extract into `Data/humanImages` folder. Rename the lfw folder as humanImages.

Download the bottleneck features for the ResNet50 model from [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) and put them into `Data/bottleneck_features/' folder.

## Project Motivation<a name="motivation"></a>


In this project, I have built a dog breed classifier that classifies dog images into their respective breeds. If the image is of a human, the classifier predicts the most resembling dog breed.

The jupyter notebook contains an in depth analysis of the dataset and the project. It iterates between different solutions before coming to the final solution.

Finally, I have created a Flask web app that allows a user to upload an image and predicts the breed.




## File Descriptions <a name="files"></a>

To run the web app, first run `train.py` in order to train the model and save the model for future prediction.

Finally, to run the web app simply type `flask run` into the terminal.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The data was provided by Udacity. Otherwise, feel free to use the code here as you would like! 


