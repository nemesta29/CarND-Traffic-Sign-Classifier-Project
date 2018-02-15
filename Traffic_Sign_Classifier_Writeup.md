
# **Traffic Sign Classifier Project** 

## Writeup

### This writeup shall provide an insight on the approach and thought process that went into making this project

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Visualisation.jpg "Visualization"
[image2]: ./index.png "Sample"
[image3]: ./Grayscale.jpg "Grayscale"
[image4]: ./Normalise.jpg "Normalised"
[image5]: ./Transformed.jpg "Trnsformed Image"
[image6]: ./1.jpg "Traffic Sign 1"
[image7]: ./2.jpg "Traffic Sign 2"
[image8]: ./3.jpg "Traffic Sign 3"
[image9]: ./22.jpg "Traffic Sign 4"
[image10]: ./5.jpeg "Traffic Sign 5"
[image11]: ./6.jpg "Traffic Sign 6"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is -> 34799
* The size of the validation set is -> 12630
* The size of test set is -> 4410
* The shape of a traffic sign image is -> 32 x 32 x 3
* The number of unique classes/labels in the data set is -> 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is divided amongst the given classes. This way we can see how well each class is represented in the dataset.

![alt text][image1]

### Design and Test a Model Architecture

#### 1.The following points shall describe how I handled the dataset and the preprocessing steps taken to ease the classification process

* As a first step, I viewed the data I was dealing with. It was a 3 - channel colored image as shown below

![alt text][image2]

* To make the process more uniform, I decided to convert the image to grayscale using the following formula -> *np.array([[0.2989],[0.5870],[0.1140]])*. I felt that for traffic signs, due to the high contrast in visibility, grayscale would give a decent representation of the image as shown below

![alt text][image3]

* As the last step, I normalized the image data to fix the data around a closed valueset and ensure that learning was smoother. I used *(image-128)/128* to normalise the pixel data containing pixel values from *0-255*, to create a normalised image as shown below

![alt text][image4]

* When I combined grayscaling and normalisation, the following image was obtained which was used as input to my classifier. The dimensions of the image so used were -> *(32x32x1)*

![alt text][image5]


* *To avoid overfitting and for a more thorough learning on the classification rules, I explored augmenting the data-set. I chose rotation, scaling, translation and flipping of the image as operations to change my existing data.This was done because there were classes with images below 400 and even 300 in number while others had data in thousands, which if left unchecked would cause learning to happen better over the well represented classes and worse on others. However, through dropouts, I was able to handle the varied representation of data and left augmentation as a future task*


#### 2. The table below shall describe the sequence and dimensions of layers used in my final model

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image						| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flattening			| Output -> 400									|
| Fully Connected 1		| 400 input, outputs 120    					|
| Fully Connected 2		| 120 input, outputs 84					    	|
| Dropout Layer			| keep_prob  *0.5* for training and *1* for test|
| Final Output			| 84 inputs, 43 outputs        					|
 


#### 3. The type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate and keep_prob have been elucidated below

To train the model, I used an Adam Optimizer for a batch size of 128. I ran it for 80 epochs, which allowed me to gain a high learning rate. Further, I tried dropouts at various layers and through trial and error, concluded that a dropout after *Fully Connected Layer 2* gave me the best results. I used a learning rate of 0.001 with a keep_prob of 0.5 for training and 1 for testing.

#### 4. Thhe Approach taken to implement my model

I chose a LeNet-5 architecture to implement my model. The following steps were taken as part of an iterative process

* Initially, the model was run for a 3 - channel RGB input for about 10 epochs. The batch size taken was 128 and a learning rate of 0.001 was chosen.Using the reLU activation, this model yielded very low results in the region of ~*55-62%* accuracy
* The first attempt was made to increase epochs to see if learning improved. I changed batch size simultaneously as well but it changed the accuracy erratically. So I kept the batch size fixed and went on increasing epochs to get an accuracy of ~*90* at roughly 80 epochs
* I then applied grayscaling and normalisation on my images, getting an input of dimensions *32x32x1*. This improved the accuracy by a few shades taking it upto the range of ~*92-93%*. However, for on newer data, it performed with low accuracy and incorrect predictions
* I experimented with dropouts, adding dropouts after each fully connected layer. This dramatically reduced my training accuracy. Instead, I began playing with dropout after each layer and saw that the best accuracy was given with a dropout after fully connected layer 2. This also improved the performance on newer images and I was happy with the results

The final model results were:
* Training and validation set accuracy of -> 94.3 %
* Test set accuracy of -> 93.9 %
* Test images accuracy -> 83 %

### Test a Model on New Images

#### 1. Six German traffic signs found on the web were chosen for testing my model

Here are six German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10] ![alt text][image11]

The points below discuss how each image tests the model differently
* The first image is placed at a skewed angle
* The second image includes other objects which resembles a natural situation
* The third image is again placed at a skewed angle which distorts its shape
* The fourth image contains a fence and a watermark which create noise
* Using the watermark as noise, the fifth image has a darker label present
* The sixth image presents a distorted shape and skewed angle along with a watermark (as noise)

#### 2. Traffic SIgn Predictions on new images

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| No Entry     			| No Entry 										|
| Priority Road			| Priority Road									|
| 30 km/h	      		| 30 km/h					 				    |
| Double Curve			| Dangerous curve on right						|
| 60 km/h	      		| 60 km/h   					 				|

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. This compares favorably to the accuracy on the test set of 93.9%. However, improvements can be made to the data to read and predict the Traffic Signs with better accuracy

#### 3. Top 5 softmax probabilities for downloaded images alongwith predictions

The code for making predictions on my final model is located in the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.88), and the image does contain a stop sign. The top five soft max probabilities of the remaining images were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .88         			| Stop sign   									| 
| .89     				| No Entry 										|
| .93					| Priority Road									|
| .90	      			| 30 km/h					 				    |
| .85				    | Slippery Road      							|
| .08				    | Double Curve           						|
| .97				    | Slippery Road      							|
