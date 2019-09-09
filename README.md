# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/center_2019_04_02_22_34_37_117.jpg "Center Camera"
[image2]: ./writeup_images/left_2019_04_02_22_34_37_117.jpg "Left Camera"
[image3]: ./writeup_images/right_2019_04_02_22_34_37_117.jpg "Right Camera"
[image4]: ./writeup_images/center_2019_04_02_22_34_37_117_flipped.jpg "Center Flipped"
[image5]: ./writeup_images/hist.png "Histogram of Steering Angles"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filters at 2x2 strides and 3x3 filters at 1x1 strides. Depths are between 24 and 64 (model.py lines 102-106) 

The model includes RELU layers for each convolution to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 101). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 107 to 115). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 96, 122, and 123). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 128).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving from both tracks, as well as center lane driving in reverse for both tracks.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start off with [Nvidia's end to end learning architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) and refined it for the simulation.

My first step was to use a convolution neural network model similar to Nvidia's end to end learning architecture. I thought this model might be appropriate because it deals with a similar task like the one for this project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To check the result, I ran the simulator to see how well the car was driving around track one. Suprisingly, the vehicle was able to complete track one and even track two. In fact, the overfitted model performed better than a "better-fitted" model with a higher mean squared error on the training set and a lower mean squared error on the validation set. However, when I experimented with another overfitted model, the vehicle was falling off track 2. 

I believe this is due to the random splitting for the training and validation sets. If the training set contained rare images for difficult situations, then the model would perform better if it was trained for those situations more (lower mean squared error on the training set but possibly higher mean squared error on the validation set). Another consideration is that, since this is a simulator, if center lane driving is maintained, the variation in images between each run will be minimal. Thus, overfitting with more epochs or splitting more data towards training actually helped the model to perform better in the simulator.

Now to combat the overfitting, I modified the model so that there is dropout between each fully-connected layer. Since the fully-connected layers are intented to be the controller for steering, I am essentially tuning the controller to be more robust.

Attempting to get better training set, I tried to basically read the CSV files 3 or 5 times, which I believe would be similar to driving and getting data for 3 or 5 times. However, it did not improve the performance at track 2. I took a look at the histogram for the steering angles, and discovered that a large numbers of steering angles corresponded to driving straight. I did not get to implement this, but I believe a method to remove some straight driving data or increase the amount of turning data (perhaps by reading CSV files for track 2 multiple times) would help generalize the model.

At the end of the process, the vehicle is able to drive autonomously around track 1 without leaving the road, using the data as is.

#### 2. Final Model Architecture

The final model architecture (model.py lines 98-115) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 color image   						|
| Cropping         		| 64x320x3 color image   						|
| Normalization    		|    											|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 30x158x24 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 13x77x36 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 5x37x48 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 3x35x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 1x33x64 	|
| RELU					|												|
| Flatten				| outputs 2112									|
| Dropout				| 0.5 keep prob 								|
| Fully connected		| outputs 100									|
| Dropout				| 0.5 keep prob 								|
| Fully connected		| outputs 50									|
| Dropout				| 0.5 keep prob 								|
| Fully connected		| outputs 10									|
| Dropout				| 0.5 keep prob 								|
| Fully connected		| outputs 1										|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then used the left and right camera images to approximate vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when needed. These images show left and right camera images:

![alt text][image2]
![alt text][image3]

Then I repeated this process on track two in order to get more data points. I also recorded data for driving in reverse for both tracks.

To augment the data sat, I also flipped images and angles thinking that this would help generalize the model. For example, here is an image that has then been flipped:

![alt text][image4]

As commented above, after obtaining these data, I found the distribution of the steering angles to be dominated by straight line driving. The 0 value from the center camera and +/- 0.2 values from the left and right cameras occured much often than other steering angle values, as this image shows:

![alt text][image5]

Regardless, I put 20% of the data into a validation set. Using a generator, I randomly shuffled the data set and created batches.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 14 as evidenced by the low training and validation losses. I used an adam optimizer so that manually training the learning rate wasn't necessary.
