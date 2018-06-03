
# lvl5 Stop Sign Heading Problem

## Challenge

Predict the heading (in degrees) of the stop sign (relative to the camera).

## Dataset

We've provided a total of 4830 images for training, testing and validation.  The image set spans the range of [-80,80] degrees of heading at increments of 1 degree, with 30 images sampled at each heading..  A sign has a relative heading of 0 degrees, when it is directly facing the camera.

Images are located in `images/sign_<HEADING_DEGREES>_XX.jpg`

#### Email George Tall <george@lvl5.ai> for any and all questions.


## My Solution

### Machine Learning Approach
#### Splitting the dataset (*split_data*)
we start by separating the dataset with the following ratios (that can be tuned): 60% for the training set, 20% for the validation set, and 20% for the testing set.
For each heading, corresponds 30 images, from which I randomly sampled the corresponding number of train, validation, and test images.

#### Data visualization
The size of the images (275 x 275 = 75,625 pixels), it is feasible to vectorize each of them and store them into a matrix. Before that, we take care to convert the images into grayscale and blur them, by performing cross-correlation with a gaussian kernel (sigma = 2) that enables us to remove the noise in the image background. That way, we constitute, X_train, X_val and X_test.
X_train is a 2,898 x 75,625 matrix.

Using Principal Components Analysis, we are able to perform a first 2D visualization of the data, that gives the following result:

![Alt text](plots/pca_2d.png?raw=true "2D visualization after PCA")

Let's look at the variance that wa can preserve while projecting on the principal loading vectors:
![Alt text](plots/pca_explained_variance.png?raw=true "Preserved variance of the first components")

Chosing k=5 components seems like a good choice, and enables us to preserve 70% of the variance of the data.

#### Transforming and fitting the training data
We then transform the training data by projecting the vectorized image vectors on the space spanned by the 5 principal loading vectors. We obtain three matrices X_train, X_val and X_test that we are going to use to train a model, validate to tune hyperparameter if needed and test it on a new source of data to obtain an accurate estimation of the performance of the model.

We choose a Nearest-Neighbor Regression algorithm as the machine learning regression model. In fact, looking at the 2D-projection, we notice that the data remains far away from one to another when the heading is sensibly different, but the datapoint of the same heading are located in a small area.

We choose a first model hyperparameter, that is the number of nearest neighbor to be equal to 3 and we fit the model with the training projected data.


#### Prediction and Performance Evaluation
Computing the predictions on the validation and test data, we obtain the following results.
The performance metrics chosen are Mean-Square Error MSE (or Root MSE), and the coefficient of determination R^2:

    Evaluation on the training set:
        Mean Squared Error: 0.012767425810904072
        RMSE: 0.11299303434683074
        R^2: 0.9999940891547172
    
    Evaluation on the validation set:
        Mean Squared Error: 0.012422360248447204
        RMSE: 0.11145564251507056
        R^2: 0.9999942489072924
    
    Evaluation on the test set:
        Mean Squared Error: 0.013457556935817806
        RMSE: 0.11600671073613718
        R^2: 0.9999937696495668

Let's look at the prediction accuracy, as if the problem was considered as a classification problem to have another observation of the performance of the model. Here are the results:

    Accuracy of the training set: 98.9%
    Accuracy of the validation set: 98.8%
    Accuracy of the test set: 98.4%



#### Conclusion
We observe that the predictions are really close to the true headings for both the test and validation set.
Moreover, the similar results between the training set, the validation set and the test set show that we did not overfit the data, because, the error are pretty similar and really low.
The RMSE on the test data shows that in average, we make mistake on the heading of the order of 0.12° which is low.
Even consider as a classification problem, we observe that the number of exact predictions is close to 100%.

Preprocessing the data to compute the 5 loading vectors and project the vectorized image on them make the computation time low at runtime, compared to keeping the entirety of the data. We were able to do that because of the principal component analysis preserving most of the variance of the data even with a few components (5 components here).

Also note that we did not use the validation set to perform hyperparameter optimization here. We could have used it to select the optimal number of nearest neighbor for example. But 3 showed good enough results and still make us able to keep a low computation runtime.

#### Test on Real Data
However, this model is based on clean images presenting only noise on them. Let's see how it deals with a real image where the background has not being removed or where there could be occlusions ...
Let's perform our algorithm on this following image:



![Alt text](real_images/1.jpg?raw=true "Real image of a stop sign")

We don't know the true heading of this image. But our machine learning model, after transforming it in grayscale, blurring, vectorizing, projecting on the first 5 components previously computed, and predicting the heading based on the 3-NN fitted model returns a prediction of a -78°, which is far from corresponding to the true heading (which is greater than 0).
Even increasing the number of nearest neighbor show a maximum heading prediction of  6° which corresponds to this orientation:

![Alt text](real_images/sign_6_0.jpg?raw=true "Stop Sign with a 6° heading")

Here is the plot showing the influence of the number of nearest neighbors considered:

![Alt text](plots/on_real_image.png?raw=true "Influence of the number of Nearest Neighbors")


This shows that our model overfit to some clean data and cannot be used as it is on not pre-processed images.
As the next steps considered, one option could be to detect the stop sign and perform image processing or segementation algorithm to remove the background. And then apply our model to predict the heading.



### Computer Vision Approach
This approach intends to make the previous solution more robust to different and more realistic images.










