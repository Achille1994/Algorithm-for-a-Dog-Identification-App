# Project Overview
Welcome to the dog breed classifier project. This project uses Convolutional Neural Networks (CNNs)! In this project, I will learn how to build a pipeline to process real-world, user-supplied images. Given an image of a dog, your algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

## About Dataset

1. The Dogs dataset contains images of 133 breeds of dogs from around the world and 8351 total dog images. The original of dataset can be Download here [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). 

2. We use OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images. The human face original dataset can be Download here [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). 

## Write an Algorithm for a Dog Identification App
The goal is to classify images of dogs according to their breed. For this project, Dog Identification App will accept any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. Example of output :

<img width="324" alt="Screenshot 2022-10-19 at 21 54 35" src="https://user-images.githubusercontent.com/74813723/196792982-4bc400dd-3e21-4075-93d9-e7af21c6476c.png">

# The Road Ahead
We break the notebook into separate steps :
- Step 0: Import Datasets
- Step 1: Detect Humans
- Step 2: Detect Dogs
- Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
- Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)
- Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)
- Step 6: Write Algorithm
- Step 7: Test Algorithm
- Step 8 build the command line application (dog_app)

Step 0: Import Datasets
- we import a dataset of dog images. We populate a few variables through the use of the load_files function from the scikit-learn library:
- train_files, valid_files, test_files - numpy arrays containing file paths to images
- train_targets, valid_targets, test_targets - numpy arrays containing onehot-encoded classification labels
- dog_names - list of string-valued dog breed names for translating labels

Step 1: Detect Humans
- We use OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images. OpenCV provides many pre-trained face detectors, stored as XML files on github. We have downloaded one of these detectors and stored it in the haarcascades directory.
<img width="308" alt="Screenshot 2022-10-19 at 22 06 51" src="https://user-images.githubusercontent.com/74813723/196794424-eb061a52-d707-4204-925a-ec965dec87ee.png">

<a id='step2'></a>
Step 2: Detect Dogs

- We use a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images.  Our first line of code downloads the ResNet-50 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.

- Write a Dog Detector
- While looking at the <a href="https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a">dictionary</a>, you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from 'Chihuahua' to 'Mexican hairless'. Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained ResNet-50 model, we need only check if the ResNet50_predict_labels function above returns a value between 151 and 268 (inclusive).

Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
- Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images. In this step, I will create a CNN that classifies dog breeds. I must create my CNN from scratch in order to compare the performannce of my model with transfer learning model.

<a id='step4'></a>
Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning VGG-16)
- To reduce training time without sacrificing accuracy, I will show you how to train a CNN using transfer learning. 

<a id='step5'></a>
## Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

I will now use transfer learning to create a CNN that can identify dog breed from images.  In this section, I must use the bottleneck features from a different pre-trained model. (I took [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features for my project).
- [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
- [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
- [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
- [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features
Because we have classification problem and we have approximately balance dog categories, we use accuracy score as metrics to evaluate performance of model. 

Step 6: Write your Algorithm
Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither. Then,

- if a dog is detected in the image, return the predicted breed.
- if a human is detected in the image, return the resembling dog breed.
- if neither is detected in the image, provide output that indicates an error.
-  please see my dog_app file for algorithm

Step 7 & 8: Test Your Algorithm
- We will test our algorithm with new image. What kind of dog does the algorithm think that you look like? If you have a dog, does it predict your dog's breed accurately? If you have a cat, does it mistakenly think that your cat is a dog? please you can app below to check the result


## Assess the Dog Detector¶
Question: 

- What percentage of the images in human_files_short have a detected dog?
- What percentage of the images in dog_files_short have a detected dog?

Answer:
- percentage of the first 100 images in human_files have a detected dog is : 0.0%
- percentage of the first 100 images in dog_files have a detected dog is : 100.0%

## Assess the Human Face Detector
Question : 

- What percentage of the first 100 images in human_files have a detected human face?
- What percentage of the first 100 images in dog_files have a detected human face?

Answer:
- percentage of the first 100 images in human_files have a detected human face is : 100.0%
- percentage of the first 100 images in dog_files have a detected human face is :11.0%

Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face. You will see that our algorithm falls short of this goal, but still gives acceptable performance. We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays human_files_short and dog_files_short.

 # web app : deploy localy
 Starter Code
The coding for this project can run in IDE environement. Here's the file structure of the project:

 


Running the Web App from the Project Workspace IDE. Here is how to see your Flask app. Open a new terminal window. go to app folder be in the workspace.
Type in the command line: python flaskapp.py ( to deploy webapp with heroku you need the comment last line begin by "app" in the file flaskapp.py)

Result of my web app below :

<img width="1186" alt="Screenshot 2022-11-25 at 22 24 08" src="https://user-images.githubusercontent.com/74813723/204060526-70aa5452-1c94-4344-8180-b93da9dcb6dd.png">
<img width="1282" alt="Screenshot 2022-11-25 at 22 25 05" src="https://user-images.githubusercontent.com/74813723/204060534-643f0fbf-5d38-4651-8080-f3465d5b8ff6.png">
