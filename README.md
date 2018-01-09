# **Traffic Sign Recognition**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./images/visualization.png "Visualization"
[image2]: ./images/unprocessed_traffic_signs.png "Unprocessed"
[image3]: ./images/yuv_ynorm_traffic_signs.png "Normalized"
[image4]: ./images/inception.png "Inception"
[image5]: ./images/acc_loss.png "Accuracy and Loss"
[image6]: ./images/GermanTrafficSigns.png "German Traffic Signs"
[image7]: ./images/softmax1.png "Softmax 1"
[image8]: ./images/softmax2.png "Softmax 2"
[image9]: ./images/softmax3.png "Softmax 3"
[image10]: ./images/softmax4.png "Softmax 4"
[image11]: ./images/softmax5.png "Softmax 5"
[image12]: ./images/softmax6.png "Softmax 6"
[image13]: ./images/softmax7.png "Softmax 7"
[image14]: ./images/softmax8.png "Softmax 8"
[image15]: ./images/st1_output.png "ST1 Output"
[image16]: ./images/st2_output.png "ST1 Output"
[image17]: ./images/inceptionL3_output.png "InceptionL3 Output"
[image18]: ./images/inceptionL4_output.png "InceptionL4 Output"
[image19]: ./images/inceptionL5_output.png "InceptionL5 Output"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/niktolis/SDCar-P2-TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **12630**
* The size of test set is **4410**
* The shape of a traffic sign image is **32x32x3**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

There are **43** individual classes in our dataset. Below there is an image of a merged histogram which included the information of how many images there are in <span style="color:red"> *training* </span> and <span style="color:blue"> *validation* </span> sets for each class. The difference in the amount of images between each label is obvious and we notice the trend is the same in both sets (training and validation). For the next step of the design we take into account that this distribution will affect the generalization of the model. Therefore we either have to augment the training dataset as some papers suggest or we use other proposed techniques e.g. spatial transformers to enhance generalization. Since data augmentation is rather straight-forward process I decided to proceed in using the spatial transformer networks as they are described in this [paper](https://arxiv.org/abs/1506.02025v3). Further description may be found on the design section.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


During the investigation of the design I tried various implementation regarding normalization. In the project code the reader may find the different approaches implemented as helper functions. The final decision that gave the best result was to change the colorspace of the images to YUV colormap with global contrast normalization across only Y channel as it was the one that it was affected the most by the light conditions. The idea of such normalization was taken by this [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).

Here as the screenshots of the images **before** the processing.

![alt text][image2]

And **after** the YUV colorspace change and Y channel normalization.

![alt text][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The model architecture is mostly inspired by [Haloi's paper](https://arxiv.org/pdf/1511.02992.pdf), with some differentiations e.g. in preprocessing method and in the amount of spatial transformers. The ST implementation can be found in [Tensorflow models](https://github.com/tensorflow/models/blob/master/research/transformer/spatial_transformer.py). The localization networks as well as the inception layers are shown below:

**Main Network**

The main network takes the preprocessed images as input and applies the first spatial transformation using a localization network which is described further below. Then there are some convolutional layers and the implementation of inception layers as described on Haloi's paper. The classification output is done using softmax. The rough architecture is shown on the table:

| Layer         		  |     Description	        					|
|:-------------------:|:---------------------------------:|
| Input         		  | 32x32x3 YUV image   							|
| ST1     	          | locNet1, outputs 128x128x3 	      |
| Convolution 5x5     | 2x2 stride, outputs 64x64x64      |
| RELU					      |												            |
| Max pooling	      	| 2x2 stride, outputs 32x32x64 		  |
| Convolution 3x3	    | 1x1 stride, outputs 32x32x192			|
| RELU                |                                   |
| Max pooling         | 2x2 stride, outputs 16x16x192     |
| ST3                 | locNet3, outputs 16x16x192        |
| InceptionL3a         | inception3a, outputs 16x16x288     |
| ST3                 | locNet3, outputs 16x16x288                    |
|inceptionL3b         | inception3b, outputs 16x16x480                   |
| InceptionL4         | inception4, outputs 8x8x832       |
| InceptionL5         | inception5, outputs 4x4x1024      |
| Avg. pooling        | 4x4 stride, outputs 1x1x1024      |
| Fully connected 1	  | inputs 1x1x1024, outputs 1024     |
| Fully connected 2   | inputs 1024, outputs 32           |
| Softmax				      |         									        |



**Localization Network**

The general localization network model is the same for all 3 STs. The output is always the same, the transformation table that is going to be applied on the features which is a 2x3. There are some minor deviations from locNet1 to locNet3 that are described on the table.

| Layer         		|     Description	        					        |
|:-----------------:|:-----------------------------------------:|
| Input         		| Depends on the layer 							        |
| Convolution (5x5 locNet1, 3x3 locNet3)    	| 2x2 stride      |
| RELU					    |												                    |
| Max pooling	      | 2x2 stride only on locNet1                |
| Convolution (5x5 locNet1, 3x3 locNet3)    | 2x2 stride locNet1, 1x1 stride locNet3	|
| RELU              |                                           |
| Max pooling       | 2x2 stride                                |
| Fully connected	1	|         									|
| Fullu connected 2 | outputs theta 2x3         |


**Inception Module**

The inception module can be visualized in the following picture taken by Haloi's paper

![alt text][image4]

The inception modules are combined in layers as described above in **main network**.


The model weights can be found [here](./model) and once the model is built they can be loaded to provide the results of the project on the test dataset once again.



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used a minibatch size, as it was advised, of 20 samples. The optimizer that was chosen finally was the ADAM. There were some learning tries with other optimizers provided by Tensorflow like RMSProp but the results were not as good as with ADAM. The starting learning rate was set to *0.00032* with a progressive decay which was indicated by the number of epochs as it is described in the table below:

| Epochs     |   Learning Rate |
|:-----------:|:---------------:|
| 0 - 25     |   0.00032        |
| 25 - 40    |   0.00015        |
| 41 - 65    |   0.00010        |
| 66 - 85    |   0.00007        |
| 86 - 95    |   0.000035       |
| 96 - 100   |   0.000015       |


 The number of epochs was 100 but the maximum validation accuracy was reached on epoch 90. Below the graph shows the progress of the final training regarding validation accuracy and loss.

![alt text][image5]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **100%**
* validation set accuracy of **98.6%**
* test set accuracy of **98.67%**

* The architecture was chosen as an initial idea derived from a very good paper which was trying to solve the same problem. Some of the concepts the paper was handling were new to me and I was working towards decrypt unknown terms like Spatial Transformers and Inception
* The initial problems were on the implementation of the chosen architecture. How to create from scratch the network that was roughly described on the paper but included many vague points due to lack of initial knowledge. Many open points were clarified after research on the web.

* From the initial architecture I removed mostly one spatial transformer which every time it was on the model training was worse compared to the training with that removed (ST2). I also included some batch normalization and dropout ready functions provided from tensorflow to reduce overfitting.

* During training I tuned the percentages of dropout and the learning rate and the way it decays over time.

* The dropouts in the localization networks and inception layers helped to greatly reduce initial overfitting of the model.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are some new traffic signs from Germany. Some of them are from the web but some of them are shot by me during my drive from home to work in order to have a bigger and more general sample

![alt text][image6]

The quality of the images from the web seem that they would be easily classified because they are clear and in the middle of the image. However the ones shot from my car are random positioned and the light conditions vary. Let alone the fact that some of them are even covered with stickers which may be add an extra difficulty to the correct prediction from the model.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to correctly guess all the given traffic signs, which gives an accuracy of **100%**. This compares favorably to the accuracy on the test set of **98.67%** given the amount of images. The images were taken from various type of "groups" as the group of speed limits or the "blue signs". It seems that the model is robust on recognizing images taken from cell phone camera from inside the car on the move.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The results regarding the softmax probabilities can be found in the following pictures:

![alt text][image7]

![alt text][image8]

![alt text][image9]

![alt text][image10]

![alt text][image11]

![alt text][image12]

![alt text][image13]

![alt text][image14]

We see that for all images the lowest correct probability is greater than 95% which is scored in the picture we were suspecting the **Keep Right** sign with a huge sticker on it. But even the rest probabilities are given to signs of the same group for all the signs. This means that that the the network is already very robust at detecting the high abstraction of the sign and struggles a little bit on the details. It is still affected a little bit by the light conditions which means that with a better preprocessing or better normalization inside the model we could achieve better results.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

Below we see some images from the internal layers of the neural network. We identify how the network is trying to find some edges as it progresses through the layers. It is because of rather big amount of features that it is not so clear in the pictures in the report. Unfortunately only the first 48 features of each layer included otherwise the pictures would be swarmed from tiny maps for each layer. It is clear though how the first spatial transformer found a table which tries to bring the image to the center of the picture by zooming on it.

**ST1 output**
![alt text][image15]
**ST2 output**
![alt text][image16]
**Inception L3 output**
![alt text][image17]
**Inception L4 output**
![alt text][image18]
**Inception L5 output**
![alt text][image19]
