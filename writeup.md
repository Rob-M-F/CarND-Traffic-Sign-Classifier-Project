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

[Label Distribution]: ./visual/raw_label_dist.png "Label Distribution"
[Normalization Sample]: ./visual/norm_sample_image.png "Normalization Sample"
[Cat 01]: ./visual/cat_01.png "Traffic Sign 1"
[Cat 04]: ./visual/cat_04.png "Traffic Sign 2"
[Cat 07]: ./visual/cat_07.png "Traffic Sign 3"
[Cat 20]: ./visual/cat_20.png "Traffic Sign 4"
[Cat 21]: ./visual/cat_21.png "Traffic Sign 5"
[Cat 27]: ./visual/cat_27.png "Traffic Sign 6"
[Normalized_new_signs]: ./visual/norm_new_signs.png "Normalized New Signs"

## Rubric Points

#### Submission Files:
The project notebook can be found at [Project Notebook](https://github.com/Rob-M-F/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) it is also available in HTML form at [Project Web-page](https://github.com/Rob-M-F/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)

#### Dataset Summary:
The German Traffic Sign dataset provides 34,799 training samples, 4,410 validation samples and 12,630 testing samples. These samples are unevenly spread over 43 sign categories. Each sample is an RGB image measuring 32 pixels on a side with an associated label. Below is a histogram of this distribution, by sample set.  

#### Exploratory Visualization:
![alt text][Label Distribution]
Distribution of labels in provided datasets.  

#### Preprocessing:
To assist with classification, I converted each image from RGB to Grayscale to reduce the effects of lighting differences and focus the model on shapes rather than colors. In addition, I normalized the pixel values from UINT8 (0 to 255) to FLOAT32 (-1 to 1), reducing the pixel value range the convolutional model needs to handle.   
![alt text][Normalization Sample]
A sample sign, pre- and post-normalized.  

Finally, I one-hot encoded the labels into a sparse array to keep the model from incorrectly assuming the sign label numbers imply structure or order.

#### Model Architecture:
My model is patterned after the LeNet architecture. It passes through 4 blocks of processing.  
The first block passes a 5x5 convolutional kernel over the image, before passing the 28x28x12 result through a leaky relu activation, providing the benefits of relu activation without the risk of dead neurons. The processing block then performs a 2x2 max pooling operation on a stride of 2, cutting the data down to 14x14x12. The second block repeats the first, bringing the output of the convolutional processes to 5x5x24.  

The model then flattens the image into a single row which is passed to a fully connected layer, cutting it down from 600 to 120 units. This passes through further leaky relu and dropout layers before the final fully connected layer reduces the data to 1 output per possible layer.

#### Model Training:
For training, I found that smaller batches outperformed larger batches consistently. As a result, my model provides batches of 43 (the number of labels) samples. The training batches are created in sets resembling the size of the original training set, which are passed through 20 training epochs at a 0.001 learning rate. To improve performance, the training batches are randomly regenerated every epoch.  

#### Solution Approach:
I attempted both the LeNet model and a GAN Discriminator model in attempting to solve this challenge. Each showed strengths and weaknesses. While the the LeNet based model had the best success on validation accuracy, the GAN Discriminator reached 100% training accuracy within the first epoch.  
My attempts to improve performance can be seen in my make_batch, perturb_image and perturb_label functions.  

Make Batch breaks datasets down into chunks that fit into the 4 gb limit of the GPU I am using. If it receives a False value for Balance, it divides up the dataset and performs no further processing. This is used for the Validation and Testing sets. 

When balance is true, it first divides the dataset up into a dictionary, with each label as the key for a list of all samples with that label. In addition, it randomly generates a matrix of desired labels measuring num_batches x batch_size. For each label in the generated matrix, Make Batch chooses a random example of that label from the dictionary. This provides a balanced training dataset for the model, returing validation accuracies of ~0.9. 

To reach the targeted accuracy of 0.93, I perturbed the images and labels. For labels, with probability 1 - perturb_label, I replaced the label with one chosen at random. For images, with probability 1 - perturb_image, I rotated the image up to 30 degrees in either direction, flipped the image along a vertical axis, added or subtracted random values from the pixels. This process also was designed to randomly repeat.  

At the start of each epoch, the entire Make Batch process was completed, generating a new set of training batches for that epoch. Using this method and the LeNet based model, validation accuracy reached 0.958.

#### Acquiring New Images:
![alt text][Cat 01]
![alt text][Cat 04]
![alt text][Cat 07]
![alt text][Cat 20]
![alt text][Cat 21]
![alt text][Cat 27]

When imported, selected images showed coloration differences from the original dataset, making grayscale a fortutous normalization choice. Each sign is readily identifiable to the eye, though each had to be significantly cropped prior to import into the model. Care was taken to vary the types of signs chosen.  

![alt text][Normalized_new_signs]

#### Performance on New Images:
On the new images, the model only achieved 50% accuracy, correctly identifying 1 of the three circular signs and 2 of the three triangular signs. This is lower than the 0.958 Validation and 0.929 Testing accuracies.  

#### Model Certainty - Softmax Probabilities:
The model correct on cat_01 by a margin of 0.05%. For cat_04 and cat_07, the confidence was more than 50%, confident, though wrong answer. It is again highly confident and wrong on cat_20 though it breaks pattern by being confident and right on cat 21 and to a lesser extent cat_27.  

These results indicate to me that this data may lend itself better to a two-step approach, first identifying sign type (circle, triangle, octogon, diamond, rectangle, etc...) then identifying the sign within that type. I believe this would allow the first model to focus on finding the sign and identifying the shape and the second to reading the sign to identify category. Something I believe is worth pursuing in the future.
