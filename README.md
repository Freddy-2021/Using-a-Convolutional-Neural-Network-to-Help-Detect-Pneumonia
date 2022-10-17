&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![example](images/pexels-pixabay-40568.png)



# Using a Convolutional Neural Network to Help Detect Pneumonia

**Author:** Freddy Abrahamson<br>
**Date created:** 7-1-2022<br>
**Discipline:** Data Science
<br><br>

## Overview
***
For this project, I will be comparing different neural network models. I will start with a ANN as a baseline model, and then create CNN(s), to see which is the most successful. 
We have a set of x-rays from both healthy patients, and ones with pneumonia. The goal for the model is to have the highest recall score. Effectively being able to correctly identify as many patients that have pneumonia as possible. 


## Business Problem
***
<b>Stakeholder:</b> Board of directors of a national network of hospitals.

<b>Business Problem:</b> Covid has caused a surge in emergency room visits. The hospital is looking for a way to better prioritize paients by the severity of their ailments, particularly pulmonary diseases.

<b>Proposed Solution:</b> A machine learning model that could distinguish between the xray image of a healthy patient, and of one with pneumonia, thereby helping prioritize who the doctor will see first.

<b>Solution Benefits:</b>  <ol>1. Helps save lives, and protect from more severe damage caused by the disease.<br>
                           2. non-invasive<br> 
                           3. cost-seffective<br>
                           4. no medical background necesary to run the model</ol>

## Data Understanding
***
The data was taken from Kaggle.com. There are a total of 5856 images. This includes 1583 'normal' images, and 4273 'pneumonia' images. The ratio of 'pneumonia' images to 'normal' images is about 2.7 : 1. I divided all these images between train , test, and val folders at a ratio of .8:.1:.1 respectively. I maintained the 2.7 to 1 ratio between the 'pneumonia' images and 'normal' images, for all the folders. Once how many of each image would go to each folder was established, all the 'normal', and 'pneumonia' images were chosen randomly. The primary concern with the dataset preparation would be to normalize the image values. All the values were scaled to a range between 0 and 1.


## Modeling
***
I used Keras and Tensorflow to create the models. Given that with the use of the filters, cnn(s) excel at detecting features of various sizes,I chose to use the less apt multi-layer perceptron as a baseline model. I then tried to overfit on purpose using a cnn. I began with a cnn model that has 4 activation layers for the feature extraction part,with the number of nodes for each each layer being 16,32,64, and 128 respectively. I used ReLu as my activation function for all feature detection, as well as for the classification layers. Given that this is a binary classification problem (0 for normal, and 1 for pneumonia), I used a sigmoid function for the output layer. From there, based on the results, I would either try to reduce the bias, by adding a layer, adding more nodes to existing layers, or both; or reduce the variance by increasing the filter size to improve generalizability, or add dropout layers.

## Evaluation
***
Given the importance of correctly identifying a patient with pneumonia, my primary goal was to find a model that produced the best recall scores. To this end, I was looking for a model that would produce the best bias/variance combination between the train and test data sets. I did this by creating a function best_model(), which utilizes the auc() function from sklearn.metrics. The x-axis is represented by the absolute difference between the train and test scores, while the y-axis is represented by the test scores. The higher the test score, and the lower the train-test difference, the greater the area under the curve. The function returns a dataframe with the models, and their respective test scores, sorted by their auc. The model with the highest auc is the best. The secondary goal was a model that would have a good accuracy score, which the 'best' model in fact does, with a score over 90%.

## Importing and Organizing Data
***
I will create two new folders: 'all_normal' and 'all_pneumonia', and copy all the corresponding images from the 'chest_xray' folder to these folders. I will then create a folder called 'train_test_val', with three folders inside of it:'train', 'test', and 'val'. Each of these three folders will have a 'normal', and a 'pneumonia' folder. I will randomly copy the images from the 'all_normal', and 'all_pneumonia' folders, to these three folders, with a split of 80%,10%,10% respectively, keeping the ratio between the number of 'normal' and 'pneumonia' images uniform across all three folders (stratified). The ratio of pneumonia images to normal images is 2.7:1.

The total number of validation images is:  586<br>
The total number of normal validation images is:  158<br>
The total number of pneumonia validation images is:  428<br>

The total number of test images is:  586<br>
The total number of normal test images is:  158<br>
The total number of pneumonia test images is:  428<br> 

The total number of train images is:  4684<br>
The total number of normal train images is:  1267<br>
The total number of pneumonia train images is:  3417<br> 



## Viewing an image
***
View an example of an x-ray of a healthy patient, and one with pneumonia.<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![example](images/xray_normal.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![example](images/xray_pneumonia.png)
<br><br>

# Neural Network Models

Six models were built all together. One ANN as a baseline model, and another five CNN(s). Below is a summary of the results. The datframe pictured is sorted by auc score. The model with the highest auc score is the model with the best bias-variance balance, and therefore the best model. In this case the best model is cnn_model_5, with a test score of 0.943038, and a train-test difference of 0.023813.<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![example](images/nn_models_df.png)
<br><br>

# Best Model Classification Report and Confusion Matrix:
## Best Model Classification Report:
<br>
The table below is the classification report for the 'Best Model' (cnn_model_5), based on predictions of the test data set. Some take-aways from the report:

1. My primary concern was with the ability of the model to correctly identify patients with pneumonia, followed by the accuracy score. Row 2, column 2 of the report confirms my initial evaluation with a recall of 94%, while still maintaining an accuracy of 95%. Although 100% is always the goal, I would consider these both great scores. 

2. In addition to this, we can see that both the precision, and f1-score metrics have very good scores of 89%, and 92%
   respectively.
<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![example](images/classification_report.png)
<br><br>

## Best Model Confusion Matrix:
I created a confusion matrix plot for the presentation, as well as an easy way to calculate the accuracy score. With 559 0ut of 586 images correctly classified, this model has an accuracy score of over 95%.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![example](images/conf_matrix.png)


# Project Conclusion: Possible Further Steps

1. Request funding for a larger dataset to further calibrate the model
2. Once the model is ready, we can implement it in a subset of emergency rooms, use the feedback to make more changes if necessary, and then expand its use from there.


## For More Information
***
Please review my full analysis in [my Jupyter Notebook](./student.ipynb) or my[presentation](./DS_Project_Presentation.pdf).<br>
For any additional questions, please contact **Freddy Abrahamson at fred0421@hotmail.com**,<br><br>

## Repository Structure

```
├── README.md                                    <- The top-level README for reviewers of this project
├── student.ipynb                                <- Narrative documentation of analysis in Jupyter notebook
├── Phase_4_Project_Presentation.pdf             <- PDF version of project presentation
└── images                                       <- Images used for this project
```
