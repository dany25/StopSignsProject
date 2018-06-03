
# Stop Sign Heading Problem

## Challenge

Predict the heading (in degrees) of the stop sign (relative to the camera).

## Dataset

We've provided a total of 4830 images for training, testing and validation.  The image set spans the range of [-80,80] degrees of heading at increments of 1 degree, with 30 images sampled at each heading..  A sign has a relative heading of 0 degrees, when it is directly facing the camera.

Images are located in `images/sign_<HEADING_DEGREES>_XX.jpg`



## My Solution

### Machine Learning Approach
#### Splitting the dataset (*split_data*)
We start by separating the dataset with the following ratios (that can be tuned): 60% for the training set, 20% for the validation set, and 20% for the testing set.
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
We then transform the training data by projecting the vectorized image vectors on the space spanned by the 5 principal loading vectors. We obtain three matrices X_train, X_val and X_test that we are going to use to train a model, validate to tune the hyperparameters, if needed, and test it on a new source of data to obtain an accurate estimation of the performance of the model.

We choose a Nearest-Neighbor Regression algorithm as the machine learning regression model. In fact, looking at the 2D-projection, we notice that the data remains far away from one to another when the headings are sensibly different, but the datapoint of the same heading are located in a small area.

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

Also, note that we did not use the validation set to perform hyperparameter optimization here. We could have used it to select the optimal number of nearest neighbor for example. But 3 showed good enough results and still make us able to keep a low computation time execution.

#### Test on Real Data
However, this model is based on clean images presenting only noise on them. Let's see how it deals with a real image where the background has not being removed or where there could be occlusions ...
Let's perform our algorithm on the following image:



![Alt text](real_images/1.jpg?raw=true "Real image of a stop sign")

We don't know the true heading of this image. But our machine learning model, after transforming it in grayscale, blurring, vectorizing, projecting on the first 5 components previously computed, and predicting the heading based on the 3-NN fitted model returns a prediction of a -78°, which is far from corresponding to the true heading (which is greater than 0).
Even increasing the number of nearest neighbor show a maximum heading prediction of  6° which corresponds to this orientation (also not close to the right heading):

![Alt text](real_images/sign_6_0.jpg?raw=true "Stop Sign with a 6° heading")

Here is the plot showing the influence of the number of nearest neighbors considered:

![Alt text](plots/on_real_image.png?raw=true "Influence of the number of Nearest Neighbors")


This shows that our model overfit to some clean data and cannot be used as it is, on not pre-processed images.
As the next steps considered, one option could be to detect the stop sign and perform image processing or segementation algorithm to remove the background. And then apply our model to predict the heading.



### Computer Vision Approach
This approach intends to make the previous solution more robust to different and more realistic images.

#### Ideas:
- Robustness to occlusion and other factors due to a change of the scene (brightness, scale, rotation, translation) can be brought by performing some corners detection (Harris-like process) and descriptions of those interest points (using SIFT for example). Then recovering them in other realistic images and using a machine learning model based on the given data set may lead to a good prediction of the heading. Or using a more geometric approach by computing homographies (the image of the plane containing the stop is performed by a projective transformation) thanks to at least 4 pairs of corresponding points; could help to retrieve the heading value.
- Performing some template matching by normalized cross-correlation on some specific corners of the image to retrieve their position. Use the data set to match the vectors linking the corners with a specific heading.
- Use Edges and eventually some Hough Line Transforms methods to detect caracteristic vectors corrsponding to a specific heading value (would be predict by a machine learning model).
- Use some properties of Stop Signs (color, shape, ... )

Let's look at the last option first and observe the specific features of a stop sign that could help us retrieve the heading value.
First, the color: flashy red! A color not that common in the stop signs picture background which is usually the sky, building, trees, ...
In our dataset, the percentage of red pixels (easily recovered using the first channel of the images, because of the aspect of our dataset) is strongly linked to the absolute value of the heading. The following image illustrates the percentage of red pixels in the image function of the heading.

![Alt text](plots/red_percentage.png?raw=true "Percentage of red pixels function of the heading")

We can fit easily this curve using degree 2 polynomial regression. On the test set, this would actually makes us able to retrieve the absolute value of the heading with a really good accuracy.

But this approach on a real world image would be strongly sensitive to the scale. In fact, the blue line indicates the percentage of the real image found (26.47%). This percentage would corresponds to an absolute value of heading around 70° on the fitted curve.
This image is an example of a 70° heading and as we see, this is still too much compared to the true heading of the real image considered earlier (around 30° approximately). But this solution is better than the one previously found based on the K-nearest neighbor.

![Alt text](real_images/sign_70_0.jpg?raw=true "Stop Sign with a 70° heading")


For a real image, how can we compute the percentage of red pixels in the images?
Looking only at the Red channel, and threshold it, will detect a lot of non-red pixel as well.
It can be interesting to look at the two ratio of the pixel value on the first channel with the pixel value on the second and third channel. Ideally, for a red color, this ratio should be high and similar.
Let's note *r_blue* and *r_green* those two ratio: by computing the value  ![equation]($$r_blue x r_green - 0.04(r_blue+r_green)^{2}$$). Thresholding that value allows us to recover a high number of red pixels coming from the stop sign. Let's see on the previous example:

![Alt text](plots/red_pixels_real_image.png?raw=true "Red Pixels in the real image")

Let's look at this solution on another real image containing red similar colors/

![Alt text](real_images/2.jpg?raw=true "Real image n2")
![Alt text](plots/red_pixels_real_image2.png?raw=true "Red Pixels in the real image 2")

This shows good results.
The next step would be to leverage that percentage of red pixel and use it to find the absolute value of the heading. But this would be constrained to some scale factor. To deal with that issue, I would spot the approximate center of the stop signs using the red pixel repartition, then use a scale-invariant function to evaluate the proper scale factor. And use that scale factor to update the red pixel percentage and predict the absolute value with the 2D-polynomial regression fitted model.

To detect the sign of the heading, my approach would be to work on the edges of the image and compare the length of the left edges and the right edges.

With more time, I would have enjoyed exploring an approach using Convolutional Neural network as well.


## References
- The two real images are taken from this dataset: https://github.com/mbasilyan/Stop-Sign-Detection





