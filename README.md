# Traffic Sign Recognition

## Project Description & Objective
Traffic sign recognition can help drivers in certain ways to increase the awareness of current road conditions and help imrpove safety 
by warning them to make sure the rules are not violated. For instance, recognizing speed limits, no entry signs or yield signs are 
just a few signs that can prove to be very important to keep the traffic safe. A front facing camera, scanning the road ahead, recognizes 
any sign and can take actions ranging from a warning on the instrument cluster, to even taking control of the cruising speed by 
automatically slowing down to make sure current speed isn't above limit. To make sign recognition possible, some machine learning methods 
are applied to train a model that can perform this recognition.

![](https://imgur.com/7m8id4n.gif)

## Running the Project
This project was developed using Google Colab. Colab offers a flexible environment to develop projects where they are synchronized 
with user's Google acount and safely uploaded to Google Drive. However, perhaps the best offer is that it allows users to benefit from 
Google's GPUs to boost performance upon running the projects. Make sure to enable GPU hardware accelerator by navigating to "Runtime" -> "Change runtime type" and select GPU under Hardware accelerator menu. This can save considerable amount of time especially when it comes to machine learning. That is why, this project only consists of a single Python notebook [trafficSigns.ipynb](trafficSigns.ipynb). After 
cloning this repository by running
```
git clone https://github.com/canozcivelek/traffic-sign-recognition.git
```
on the command prompt, the project is pretty much ready to go. If desired to run this project on Google Colab, make sure to have a Google account. Then by simply navigating to https://colab.research.google.com, upload the project and run all the cells. More notes on  how the code works are provided in the next section.

## How the Code Works
### Acquiring the Dataset
After launching the notebook on Colab or Jupyter, the first cell clones the necessary resources of all the traffic signs and their labels from a github repo. This is followed by importing the libraries such as numpy, matplotlib and keras. Next, data are loaded to make possible working with them and analyze them. 

### Analyzing the Dataset
Upon inspection, it is understood that all the traffic sign images have a dimension of 32x32 and 3 channels of color. However, to be able to train the model, it is needed to preprocess the images in such a way that it is easier for the training part and doesn't take too much time. signnames.csv file contains all 43 signs and their corresponding labels. For instance, "Speed Limit 50km/h" sign has number "2" as its label, or "No Entry" sign has the label "17". By plotting the number of samples, it is seen how much data is provided for each traffic sign.

### Preprocessing Images
As mentioned earlier, a preprocessing is required to prepare the dataset for training. 
* First, converting them to grayscale images will get rid of the 3-channel layout to the desired single channel layout. 
* Applying histogram equalization will evenly distribute each pixel density and standardize lighting which will make sure the images look less messy and more composed.
* Normalizing each pixel density by dividing them by 255. Normally density ranges from 0-255, after performing the division, all the densities will be ranging between 0-1.
These steps are performed under the function preprocess() and prepares the images for training.

### Data Augmentation
It is possible to augment the dataset by making a few modifications on each image. This will help generating more images to train, thus, more accurate learning rates will be achieved. Using keras.preprocessing.image library, ImageDataGenerator is imported and this library will enable data augmentation by shifting images, zooming in/out, shearing rotating them randomly.

### The LeNet Model
The LeNet model is used to perform training. This model is proved to be an efficient model and provides high rates of accuracy.

**Adding the Convolutional Layers**

In this _sequential_ model, it is first added 2 convolutional 2D layers each consisting of 60 filters which are responsible for extracting features out of the training images. These features have an essential role in **predicting** what a sign looks like and correctly classifying them. They have a 5x5 dimension that will scan through each image which are reduced to 28x28 in dimension. The next parameter defines the input image shape which were defined earlier as 32x32x1. Lastly, the layers are activated using "relu" activation function. 

**Adding a Pooling Layer**

As per the LeNet model architecture, a pooling layer is added to reduce the feature map dimensions from 5x5 to 2x2. Basically, this will prevent overfitting by having more generalized versions of previously extracted features and provide less parameters to work with.

**Further Convolutional Layers**

2 more convolutional 2d layers are added, this time, having 30 filters each with dimensions of 3x3. They are again followed by a pooling layer with a pooling size of 2x2.

**Dropout Layers**

A dropout layer is added to make some of the input nodes dropped. a 0.5 rate of dropout means at each update, half of the input nodes will be dropped. Which will speed up the process and not have a drastic effect on learning.

**Flatten & Dense Layers**

By adding a flatten layer, the data is formatted properly to be fed into the fully connected layer as a one dimensional array. Next, by declaring a dense layer, all the nodes in the subsequent layer is connected to every node in the preceding layer.

### Training & Analyzing
After defining the model, it's time for training to take place. The model is trained in 2000 steps per each epoch, and epoch size is defined to be 10. This is where Google Colab's hardware accelerator really makes a difference as it significantly reduces the time it takes to complete training. When training is complete, some visualization is done by plotting the "Loss" and "Accuracy" functions of the training. Analyzing these functions is important to make sense of how the training went. One can make adjustments on the model if these graphs show signs of overfitting, underfitting etc. It is seen here that test accuracy is over 97% which is good enough for the purposes of this project.

### Trying Out the Model
Finally, from a URL, a traffic sign is tried to see if the model predicts the sign correctly. The output is given as the predicted sign label which are defined in the signnames.csv file.








































