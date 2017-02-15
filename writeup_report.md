#**Behavioral Cloning**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

The model I picked is as described https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf. I guess this
is called the nvidia model.

See the `create_model()` method for definition of the model
* line 28 - extra normalization layer
* line 29 - cropping input images to remove unnecessary information to improve training speed.
* line 32 to 26 - 5 Convolution2D layers

####2. Attempts to reduce overfitting in the model

* Line 42 of model.py has a dropout layer to reduce overfitting.
* Model was trained on data from track 1, track 2, both driven in forwards and reverse.
* Data from all 3 sets of cameras was used.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 111). Started with a learning rate of `0.0001`

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.

* Model was trained on data from track 1, track 2, both driven in forwards and reverse.
* Data from all 3 sets of cameras angles with appropriate compensation was used.
* Also, some data from recovery from left or right was used.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to read teh project hints and follow suggestions. Suggestion to create
nvidia model was strong enough that it made sense to give it a try.

Here are the results of model training, the validation loss seemed low enough to give this trained model a try.

```
Train on 18134 samples, validate on 4534 samples
Epoch 1/10
18134/18134 [==============================] - 30s - loss: 0.0671 - val_loss: 0.0581
Epoch 2/10
18134/18134 [==============================] - 28s - loss: 0.0541 - val_loss: 0.0508
Epoch 3/10
18134/18134 [==============================] - 25s - loss: 0.0485 - val_loss: 0.0488
Epoch 4/10
18134/18134 [==============================] - 25s - loss: 0.0441 - val_loss: 0.0454
Epoch 5/10
18134/18134 [==============================] - 25s - loss: 0.0402 - val_loss: 0.0414
Epoch 6/10
18134/18134 [==============================] - 24s - loss: 0.0364 - val_loss: 0.0381
Epoch 7/10
18134/18134 [==============================] - 24s - loss: 0.0325 - val_loss: 0.0378
Epoch 8/10
18134/18134 [==============================] - 24s - loss: 0.0305 - val_loss: 0.0342
Epoch 9/10
18134/18134 [==============================] - 24s - loss: 0.0280 - val_loss: 0.0322
Epoch 10/10
18134/18134 [==============================] - 24s - loss: 0.0249 - val_loss: 0.0306
```
Fired up the simulator in autonomous mode to see how well the model performs.
* Track 1 - car seemed to properly stay in the middle. There were sometimes when it seemed to navigate
            to the edge but was able to recover.
* Track 2 - model performed poorly when it managed to drive off the road. However, model did keep car
            on the track for a atleast 1/2 the track before it made an unrecoverable error.

At the end of the process, the vehicle is able to drive autonomously around the track 1 without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 26-54) consisted of a convolution neural network with the following layers and layer sizes

This is the model summary as output by model.py line 52.

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 32, 64, 3)     0           lambda_input_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 18, 64, 3)     0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 9, 32, 24)     1824        cropping2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 5, 16, 36)     21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 3, 8, 48)      43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 8, 64)      76864       convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 8, 64)      36928       convolution2d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1536)          0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          1789068     flatten_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1164)          0           dense_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 1164)          0           dropout_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      activation_1[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 100)           0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        activation_2[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 50)            0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         activation_3[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 10)            0           dense_4[0][0]
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          activation_4[0][0]
====================================================================================================
Total params: 2,091,639
Trainable params: 2,091,639
Non-trainable params: 0
____________________________________________________________________________________________________
```

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center][samples/center.jpg]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![right_recovery][samples/right_recovery.jpg]

Then I repeated this process on track two in order to get more data points.

After the collection process, I had 7500+ number of data points. I then preprocessed this data by resize to 1/5 of original size and also cropping image data.
Also, used left, right and center images to get some more data points.

I finally randomly shuffled the data set and put 10% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10.
I used an adam optimizer so that manually training the learning rate wasn't necessary.
