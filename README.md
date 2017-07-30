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

[image1]: ./images/norm_hist.png "Normalized histogram of the data set"
[image2]: ./images/random_picture.png "BUmpy road ahead picture"
[image3]: ./images/preprocessing.png "Traffic sign before and after preprocessing"
[image4]: ./images/failed1.png "Failed prediction 1"
[image5]: ./images/failed2.png "Failed prediction 2"
[image6]: ./images/failed3.png "Failed prediction 3"
[image7]: ./images/failed4.png "Failed prediction 4"
[image8]: ./images/failed5.png "Failed prediction 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/janulicny/sdcnd-traffic-signs-classification/Traffic_Sign_Classifier-submit.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python and numpy commands to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410 
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. This histogram shows that the distribution of classes is aproximately similar across the training, validation and testing data sets. ANother thing worth mentioning is that some classes have way more examples than others. This may make the model prone the classifying the well represented classes preffereably to the ones with less data points. This could be imroved with data augmentation.

![normalized histogram][image1]

In the code, random image from training set is shown with its label. Here is a an example of *Bumpy road ahead* sign:

![random picture][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale, since I read in an comparison of that the best algorithms for this dataset that the conversion to grayscale did not hurt the performance. Also I wanted to reduce the size of the input into first layer since I was training on my laptop.

Afterthat I apllied normalization, as was suggested during the course. I used the naive approach mentioned in the code template. NExt tme, I might use the normalization function included in scipy package

Here is an example of a traffic sign image before and after preprocessing, the normalization cannot be seen, because pyplot's *imshow()* functions scales the inputs to 0-255 range.

![Traffic sign before preprocessing][image3]



I did not generate any augmented data due to time constrains, but that would surely help the performance of the model, particularly in the not well represented classes.

I would use translation, rotation, brightness augmentation and blurring to generate more data.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I have used the architecture we used in the lessons and implemented it in Keras. The  sequential model consists of two convolution layers with maxpooling, dropout and RELU activation, followed by two fully connected layers.


| Layer         		|     Description	        					| 
|-----------------------|-----------------------------------------------| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x120 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x120 				|
| Droput				| 50 %											|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x50 	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x50					|
| Droput				| 50 %											|
| RELU					|												|
| Fully connected		| outputs 128      								|
| RELU					|												|
| Fully connected		| outputs 43      								|
| Softmax				| 	        									|
 



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used *adam* optimizer [[paper](https://arxiv.org/abs/1412.6980v8)] and *categorical cross-entropy* loss function, because it was listed as an example in Keras documentation. I also specified that I want to include accuracy in metrics list.

The default parameters (same as in the original paper) of the *adam* optimizer were used.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 99.20 %
* validation set accuracy of 94.81 % 
* test set accuracy of 92.91 %

I used the architecture that was decribed in the lessons in course. On the first run, I got validation accuracy of 92.5 %. I trained the model for 10 epochs and noticed that the validation accuracy started to drop after fourth epoch, this pointed to overfitting of model. I reran the model with just 4 epochs and the accuracy was above the required 93 % on validation data set.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify. 
#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric). 
#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

During my business trip to germany, I have taking 27 photos of traffic signs.
Out of these, my model correctly classified 22 images. 

The model was able to correctly guess 20 of the 27 traffic signs, which gives an accuracy of 81.48 %. This is lower than the accuracy on the test set.

I will to go through the five failed predictions and try to figure out what might have confused the model.

* **1. failed prediction**

This *General caution* sign was misclasified as *Slippery road*, but the algorithm was not sure the top probability was only 29%. The next four guesses can be seen in the image below and they do not contain *general caution*. In this case I think the explanation is pretty straight forward - the additional info plate below the sign is something that the model did not encounter in the training set and it cannot handle it. I beleive that croping this image to contain just tje sign would allow the model to appropriately label the image.
 
![Failed prediction 1][image4]


* **2. failed prediction**

This one is more interesting since *Speed limit 80 km/h* was misclassified as *Speed limit 50 km/h* with probabilty 49 %, followed by *Speed limit 60 km/h* with probabilty 32 % and only on the third place correct label *Speed limit 80 km/h* with  18 % confidence. I was really surprised that the model was not able to correctly label this image, because it seems really easy to me. The only explanation that I can think of tha slight blur of the image. That's why I would like to include blurring in augmenation step in future improvement efforts.

![Failed prediction 2][image5]
 

* **3. failed prediction**

Again in this case, I feel that the blur of the image is responsible for the failed prediction. The *No passing sign* was clasiffied as *Speed limit 70 km/h* with 100 %.

![Failed prediction 3][image6]


* **4. failed prediction**

Here, the text *Zone* below the sign confuses the model to clasiffy this *Speed limit 30 km/h* as *Keep right*. The correct label is the second guess, so we can say that the model was at least partially sucessful in this difficult example. 

![Failed prediction 4][image7]


* **5. failed prediction**

And last, this overexposed, blurred photo of *Bicycles crossing*. I was kind of surprised to see that eventhough the model thinks that the image is of *Road narrows on the right*, the second guess is the correct one. If I didn!t remember, what photos did I take, I don't think I would be able to guess this sign myself. I was impressed.

Then I found out that the best neural network outperform humans on the german traffic signs dataset. Wow.

![Failed prediction 5][image8]
