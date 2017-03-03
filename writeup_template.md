#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric

This is a per-point addressment of the Udacity review rubric. It's sort of like filling a form instead of writing freeform text, but I hope reviewing the document is therefore easier and more structured.

A reference to the rubric is visible here: [rubric.](https://review.udacity.com/#!/rubrics/481/view)

---

### Writeup

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Hi! Thank you for reviewing my project even though it's a few days past the deadline. I needed to add some final touches in order to be satisfied with the result.

My project is included as a zipfile and also visible on GitHub: [project](https://github.com/boxmein/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)


### Data Set Summary & Exploration

In my code, each of the relevant sections have a proper heading with the same title. I won't mention this again in the below segments :)

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Some simple python and Numpy functions were used to calculate statistics.

- Number of training examples = 34799
- Number of testing examples = 12630
- Image data shape = (32, 32, 3)
- Number of classes = 43


#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The exploratory visualization includes a histogram showing the frequency of each class in the training set. It also shows a random sample of an image in the training set, just to know what we're dealing with.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

My preprocessing code normalizes the image data to have every pixel value between -1 and 1. It doesn't grayscale the images, because the colors contain a lot of information as to which class they belong to. (Stop signs are red, after all!)

I preprocessed the classes to convert them to one-hot encoding, using a simple Numpy function from Stack Overflow. Sadly I haven't worked too much with Numpy so a preimplementation was the better way out here.

![alt text][image2]

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I set up the data by importing it from the existing pickle files. The dataset sizes were 34799 for original training data, 4410 for original validation data and 12630 for original testing data.

The cross validation function I used simply used the entire validation set for every epoch. No splitting was needed if I used the 
presplit dataset.

I augmented the training data by creating 7 images per test image, with random alterations to the luminance. The luminance augmentation source has been taken from a different project, with changes to fit in my current solution, and has been linked in the respective code cell. 

My final training set had (34799 + 243593 = 278392) number of images. I did not augment my validation and test sets.

Here is an example of an original image and an augmented image:

![alt text][img1]

![alt text][img1_augmented]

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is a modification of the LeNet architecture, consisting of the following layers. I will include a graph as well.

**TODO: GRAPH HERE.**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:|
| ## Input layer  ##	|												|
| Input         		| 32x32x3 RGB image   							|
| ## Convolution 1 ##	|												|
| 5x5x3 convolution 	| Output - 28x28x6. Stride 1, padding valid.    |
| ReLU 					| Output - same size. 							|
| Max pooling			| Output - 14x14x6. Stride 2, padding valid.	|
| ## Convolution 2 ##	|												|
| 5x5x6 convolution.	| Output - 10x10x16. Stride 1, padding valid.	|
| ReLU					| Output - same size. 							|
| Max pooling			| Output - 5x5x16. Stride 1, padding valid.		|
| ## Convolution 3 ##	|												|
| 1x1x16 convolution.	| Output - 5x5x24. Stride 1, padding valid.		|
| Dropout				| It's dropout! :D  							|
| Flatten				| Reshape into a flat list. Output - 600x1		|
| ## FC 1 ##			|												|
| Fully connected layer	| A simple fully connected layer. Output - 120x1|
| ReLU					| Output - same size. 							|
| Dropout				| It's dropout! :D  							|
| ## FC 2 ##			|												|
| Fully connected layer	| A simple fully connected layer. Output - 84x1 |
| ReLU					| Output - same size. 							|
| ## Output ##			|												|
| Output layer			| Output - 43x1.								|
| ## For training: ##   |												|
| Cross validation		| Comparing real result to classifier output	|
| Adam optimizer		| Learning rate is defined in code				|

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

**TODO: WRITE**

The training code is included in 

|						|												|
|						|												|
To train the model, I used an ....

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

**TODO: WRITE**

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 