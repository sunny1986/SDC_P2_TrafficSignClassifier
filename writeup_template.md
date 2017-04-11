## Project 2: Traffic Sign Recognition

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

[image1]: ./datasetexplore.JPG
[image2]: ./hists.JPG
[image3]: ./accuracies.JPG
[image4]: ./training_history.jpg
[image5]: ./training_history_early_term.jpg
[image6]: ./web1.jpg
[image7]: ./web2.jpg
[image8]: ./web3.jpg
[image9]: ./web4.jpg
[image10]: ./web5.jpg
[image11]: ./web1_softmax.jpg
[image12]: ./web2_softmax.jpg
[image13]: ./web3_softmax.jpg
[image14]: ./web4_softmax.jpg
[image15]: ./web5_softmax.jpg

#### Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
The submission includes the project code and a write up report. What you're reading is the report and here is a link to my [project code](https://github.com/sunny1986/SDC_P2_TrafficSignClassifier/blob/master/Traffic_Sign_Classifier_Final_Solution.ipynb)

#### Data Set Summary & Exploration

##### Data Summary - CODE CELL # 2 of the IPython notebook.  

I used the pickled data set and calculated the summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3) 
* The number of unique classes/labels in the data set is 43

##### Exploration 
CODE CELL # 3 & 4 of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing a histogram of the 43 classes of traffic signs.
As we can see the training set has a good number of samples for almost first 15 classes but mostly lesser for classes thereafter. It 
gives us a hint the model will not be trained on a larger variety of samples in these latter classes and hence might have lower performance
in prediciting such classes.

![alt text][image1]

#### Design and Test a Model Architecture

##### Preprocessing 
CODE CELL # 5 of the IPython notebook.

The shape of the traffic sign data set is 32,32,3. I first analyzed the histogram of some sample images and understood in a lot of images the intensities were concentrated within in small range. Hence I equalized the histogram to maintain a good distribution of intensities over the images. Here is an example of a traffic sign image before and after histogram equalization.

![alt text][image2]

As the next step, I normalized the image data to help the optimizer in reaching the minima for the cost function quicker and also not get stuck in a local minima.

##### Data partition into test and validation set
In CODE CELL # 1 the data is read from the provided pickle files which was already divided into training set, validation set and test set. 

* My final training set had 34799 images. 
* My validation set had 4410 images
* My test set had 12630 images.


##### Model Architecture
CODE CELL # 8 of the ipython notebook. 

My final model was based on the LeNet architecture which consisted of the following layers:

| Layer         			|     Description	        					| 
|:---------------------:	|:---------------------------------------------:| 
| Input         			| 32x32x3 RGB image   							| 
| Convolution 5x5     		| 1x1 stride, VALID padding, outputs 28x28x6 	|
| Activation: RELU			|												|
| Max pooling	      		| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    	| 1x1 stride, VALID padding, outputs 10x10x16 	|
| Activation: RELU			|												|
| Max pooling	      		| 2x2 stride,  outputs 5x5x16   				|
| Fully connected			| Flattened input 400 outputs 120   			|
| Activation: RELU			|												|
| Dropout:      			| Keep prob = 0.5 								|
| Fully connected			| Flattened input 120 outputs 84    			|
| Activation: RELU			|												|
| Dropout:      			| Keep prob = 0.5 								|
| Fully connected			| Flattened input 84 outputs 43     			|
| Softmax					| Converting logits to probabilities 			|
|							|												|
|							|												|


##### Training Pipeline
CODE CELL # 7 of the ipython notebook

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I chose a batch size of 64 and a total of 250 epochs. I used the given Adam optimizer with a learning rate of 0.0009.

##### Training Approach
CODE CELL # 8 of the ipython notebook 

Plot showing training history with 250 Epochs:

![alt text][image4]

###### No Early termination

My model results without early termination:
* training set accuracy of 100%
* validation set accuracy of 97.7%
* test set accuracy of 94.7%

Steps on training and improving the model 
* The architecture was based on standard LeNet model which included 2 convolutional layers and 3 fully connected layers
* The standard LeNet architecture gave good training accuracy but poor validation accuracy. The reason was that the model was overfitting the training data and hence couldn't generalize much. 
* In order to prevent overfitting, a regularization method called dropout was introduced in the architecture which improved the validation set accuracy.
* Also in convolutional layers when filter strides thorugh the image there is a loss of information while downsampling. In order to avoid this a pooling layer was also added which helps downsample the image spatailly for the next layer.
* Initially the learning rate was at 0.01 which was quite high as the plots which mapped training and validation accuracy showed as being pretty jagged. Once the learning rate was tuned and reduced further upto 0.001 the accuracy graphs also smoothened. 
* The LeNet architecture is a good starting point for this problem since it can deal very well with translation invariance as seen in this model's successful performance in case of MNIST dataset. The traffic signs can be anywhere in image frames captured by a self-driving vehicle hence it is a good application for the LeNet model for classication.
* Also this architecture provides flexibility to add more layers amd make the network deeper in order to resolve complex problems like the problem of traffic sign classification. 

###### Early termination
Looking at the training history and to avoid overfitting, I used the model with early termination and retrained for 90 epochs instead of the 250.

![alt text][image5]

My model results were with early termination:
* training set accuracy of 99.9% (prevented overfitting)
* validation set accuracy of 97.7% (remained same)
* test set accuracy of 95.2% (slight improvement in test accuracy)


#### Test a Model on New Images

##### Here are the five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

Image 1, 3 & 5 might be tricky to identify since the signs are at a different perspective and look sheared by certain degree. The training data set that is provided has signs which are mostly straight from one's perspective and have almost no shear or rotational component relative to the image axes. Images 2 & 4 are relatively similar to the training set images. However it will be interesting to find if the last image is correctly identified or not since it is pretty close to the perspective of the training set images.

##### Predictions
CODE CELL # 14 of the Ipython notebook.

Web images accuracy = 20%
The model was able to only predict 1 out of 5 images correctly. Interestingly image 5 was not predicted correctly although it seemed similar to training images except for the slight perspective difference.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep Right      		| General caution								| 
| Children Crossing		| Keep left         							|
| Pedestrian Crossing 	| Road work             						|
| Bumpy Road	   		| Bumpy Road					 				|
| Stop Sign 			| Priority road     							|

Looking at the performance of the model on these web images and the possible features that make it difficult for the model to correctly predict, further improvements on the model can include training data augmentation like flipping, rotating, introducing noise in the images and increasing the size of the training set.

##### Softmax probabilities for web images
CODE CELL # 13 of the Ipython notebook.

###### For the 1st image:

* Actual         > Keep right 
* Top prediction > General caution

![alt text][image11]

###### For the 2nd image: 

* Actual         > Children crossing sign
* Top prediction > Keep left

![alt text][image12]


###### For the 3rd image:

* Actual         > Pedestrian crossing sign
* Top prediction > Road work

![alt text][image13]


###### For the 4th image:

* Actual         > Bumpy Road sign
* Top prediction > Bumpy Road sign

![alt text][image14]


###### For the 5th image:

* Actual         > Priority Road
* Top prediction > Stop sign

![alt text][image15]