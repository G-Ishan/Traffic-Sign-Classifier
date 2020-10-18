# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/training-visualization.png "Visualization"
[image2]: ./examples/no-entry-orig.png "Original Sign"
[image3]: ./examples/no-entry-processed.png "Grayscale Sign"
[image4]: ./examples/online-signs.png "Online Signs"
[image5]: ./examples/softmax-visualization.png "Softmax"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup 

#### This writeup includes all the rubric points and how I addressed each one.

Here is a link to my [project code](https://github.com/G-Ishan/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb) for reference

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of the feature labels in the data.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because even though color matters in traffic signs, I wanted the model to focus on other features like shape and content of the sign.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2] ![alt_text][image3]

I also normalized the image data in order to get the mean to be close to zero. This was the only two preprocessing steps I did for the images.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 |
| Fully connected		| Input 400, Output 120 |
| RELU	and dropout				|												|
| Fully connected		| Input 120, Output 84 |
| RELU	and dropout				|												|
| Fully connected		| Input 84, Output 43 |
| Softmax				|              |

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.


To train the model, I used an Adam optimizer and the following hyperparameters:

batch size: 100
number of epochs: 50
learning rate: 0.0009
mu = 0.0 and sigma = 0.1
keep probability of the dropout layer: 0.5


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.9%
* validation set accuracy of 96.2%
* test set accuracy of 94.3%

I used an iterative approach to finding the right adjustments to the model. The cell labeled "Model Changes" in the notebook contains a log of the changes I made. These changes primarily included changing the HYPERPARAMETERS and analyzing what result these changes made. I used well known ADAM optimizer and LeNet architecture because I knew these were successful in a handwriting classifier and I believed this method would translate well for classifying traffic signs.

Givent the high accuracies across all the data sets, it is shown that the model is working well.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4]

I made sure to resize these images to the same size as the original data set of (32, 32, 3). I then preprocessed (grayscale and normalize) these images before running them through the LeNet pipeline. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead Only     		| Speed Limit (80 km/h)   									| 
| Priority Road     			| Priority Road 										|
| Yield					| Yield											|
| No Entry	      		| Speed Limit (80 km/h)					 				|
| General Caution		| Speed Limit (80 km/h)      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares poorly to the accuracy of the test set at 94.3%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

To understand where the model is struggling, I plotted the top 5 softmax probabilities for each sign seen in the image below. Here, we can see in the 3 incorrect signs, the model is really sure that they are all 80 km/h speed limit signs, with other speed limit signs as the next best probabilities. The no entry sign at least has a probability of .05 for no entry but the ahead only and general caution do not even have a correct probability in the top 5. I am not sure why the model has such a bias for speed limit signs. 


![alt text][image5]



