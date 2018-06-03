#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 23:22:07 2018

@author: daniel
"""
import numpy as np
from random import shuffle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage.filters import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor


# Tool functions
def compute_red_percentage(img, treshold):
    return len(img[img>=treshold])/(img.shape[0]*img.shape[1])


# Performance Evaluation of the model    
def scoring(y_true, y_pred):
    RMSE = np.sqrt(((y_true - y_pred)**2).mean())
    MSE = ((y_true - y_pred)**2).mean()
    print("\t Mean Squared Error: {0}".format(MSE))
    print("\t RMSE: {0}".format(RMSE))
    # SCORE - R^2
    R2 = 1 - ((y_true - y_pred)**2).sum()/\
          ((y_true - y_true.mean()) ** 2).sum()
    print("\t R^2: {0}".format(R2))
    

def vectorize(images):
    """ Convert the images into feature vectors of size (width x height)"""
    X_vec = np.zeros((images.shape[0], images.shape[1]*images.shape[2]))
    for i in range(images.shape[0]):
        temp_gray_img = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
        temp_gray_img = gaussian_filter(temp_gray_img, sigma = 2)
        temp_gray_img = temp_gray_img.ravel()
        X_vec[i] = temp_gray_img
    return X_vec


class StopSigns:
    
    # Attributes of the data
    height, width, channels = 275, 275, 3
    min_heading = -80
    max_heading = 80
    num_image_per_heading = 30
    
    # Parameters to tune
    red_threshold = 200
    
    
    def __init__(self):
        """
        Attributes
        """
        self.headings = np.array(range(self.min_heading,self.max_heading+1))
        self.train_images, self.train_labels = 0, 0
        self.val_images, self.val_labels = 0, 0
        self.test_images, self.test_labels = 0, 0
        
        """
        Fitting parameters
        """
        w_red_percentage = 0
        
        
        
    
    def split_data(self, p_train = 0.6, p_val = 0.2, p_test =  0.2):
        """
        Split the data into a training, validation and testing set with the 
        corresonding percentages
        """
        
        assert p_train + p_val + p_test == 1, \
        "the ratio of the different set has to add up to 1"
        
        list_train=[]
        list_val=[]
        list_test=[]
        
        print("Splitting the data ...")
        
        for h in self.headings:
            if (h%10 == 0):
                print("\t heading: {0}".format(h))
            indices = np.array(range(0,self.num_image_per_heading))
            shuffle(indices)
            
            num_train = int(p_train*self.num_image_per_heading)
            num_val = int(p_val*self.num_image_per_heading)
            num_test = int(p_test*self.num_image_per_heading)
            
            for i in range(self.num_image_per_heading):
                new_img = mpimg.imread("images/sign_"+str(h)+"_"+str(indices[i])+".jpg")

                if i<num_train:
                    list_train.append(new_img)
                elif i< num_train + num_val:
                    list_val.append(new_img)
                else: 
                    list_test.append(new_img)

            self.train_labels = np.vstack([self.train_labels,np.full((num_train,1),h)])
            self.val_labels = np.vstack([self.val_labels,np.full((num_val,1),h)])
            self.test_labels = np.vstack([self.test_labels,np.full((num_test,1),h)])
        
    
        self.train_images = np.stack(list_train)
        self.train_labels = self.train_labels[1:]
        
        self.val_images = np.stack(list_val)
        self.val_labels = self.val_labels[1:]
        
        self.test_images = np.stack(list_test)
        self.test_labels = self.test_labels[1:]
        print("Splitting done!")
    




    def preprocessing(self, images):
        """
        Images are drawn from the training, validation or testing set.
            images is usually: self.train_images, self.val_images, or self.test_images
        Return the matrix of features drawn from the images
        """
        
        # Compute red percentage in an image
        red_percentages_ = []
        
        for i in range(images.shape[0]):
            red = compute_red_percentage(images[i],self.red_threshold)
            red_percentages_.append(red)
        X = np.array(red_percentages_).reshape((-1,1))
        
        
        return X
            
    


    
    def fit(self, X, y, plot = 0):        
        
        # compute the absolute value based on the red percentage on the image
        H = np.hstack([np.ones((y.shape[0],1)),\
                        y,\
                        (y**2)\
                    ])
        self.w_red_percentage = np.linalg.solve(H.T@H, H.T@X[:,0])
        

        if plot: 
            w = self.w_red_percentage.copy()
            plt.title("Red Percentage function of the heading fitted")
            plt.plot(self.headings, [w[0]+ i*w[1]+w[2]*i**2 for i in self.headings], c='b')
            plt.scatter(y, X[:,0], s = 1, c ='r')
            plt.show() 
            
            plt.title("ZOOMED - Red Percentage function of the heading fitted")
            plt.plot(self.headings[:10], [w[0]+ i*w[1]+w[2]*i**2 for i in self.headings[:10]], c='b')
            plt.scatter(y[:120], X[:120,0], s = 1, c ='r')
            plt.show() 
            
            
            
    
            
        
        
        

if __name__ == "__main__":
    

    
    split_done = 0
    if not split_done:
        model = StopSigns()
        model.split_data()
        
    preprocessed = 0
    if not preprocessed:
        X_train = model.preprocessing(model.train_images)
        y_train = model.train_labels
        
        y_test = model.test_labels
        y_val = model.val_labels
        
    model.fit(X_train,y_train, plot=1)
    
    
   
        
    
        
    """ SANDBOX """
    """
    # original image
    sand_img = model.train_images[150].copy()
    plt.imshow(sand_img)
    plt.show()
    # gray image - then blurred (gaussian)
    gray_img = cv2.cvtColor(sand_img, cv2.COLOR_RGB2GRAY)
    gray_img = gaussian_filter(gray_img, sigma = 2)
    plt.imshow(gray_img,cmap='gray')
    plt.show()
    
    
    # find threshold for white
    plt.hist(gray_img.ravel(), bins=100)
    plt.show()
    
    white_threshold = 145
    
    argmax_indices=[]
    for i in range(gray_img.shape[0]):
        row = gray_img[i,:]
        argmax_indices.append(np.where(row>=white_threshold)[0][0])
    argmax_indices = np.array(argmax_indices)
    
    plt.plot(argmax_indices)
    plt.show()
    
    # remove noisy background
    white_img = gray_img.copy()
    white_img[white_img<white_threshold]=0
    plt.imshow(white_img,cmap='gray')
    plt.show()
    
    
    
    """
    # Machine Learning approach
    # VECTORIZATION + PCA + K-NN

        
    X_train = vectorize(model.train_images)
        
    """ PCA and 2-D visualization """
    
    # How many components ?
    pca = PCA(n_components=10)
    pca.fit(X_train)
    print("explained_variance_ratio_: ",pca.explained_variance_ratio_)  
    plt.title("explained variance ratio function of the i_th component")
    plt.plot(pca.explained_variance_ratio_)
    plt.show()
    
    nb_components = 5
    pca = PCA(n_components=nb_components)
    pca.fit(X_train)
    X_train_proj = pca.transform(X_train)
    print("explained_variance_ratio_: ",pca.explained_variance_ratio_)   
    for i in range(161):
        plt.scatter(X_train_proj[:,0][18*i:18*(i+1)], X_train_proj[:,1][18*i:18*(i+1)], c=str(i/161.))
    plt.title("2D projection of the images after PCA")
    plt.show()
    
    
    
    
    """ K-NN on 5 principal components """
    
    neigh = KNeighborsRegressor(n_neighbors=3)
    neigh.fit(X_train_proj, y_train) 
    
    X_test = vectorize(model.test_images)
    X_test_proj = pca.transform(X_test)
    y_test_predict = neigh.predict(X_test_proj)
    y_test_predict = y_test_predict.astype(np.int8)
    
    
    # SCORE 
        
    print("Evaluation on the Test Set")
    scoring(y_test, y_test_predict)
        

    
    """ RESULTS on Test Set
    Mean Squared Error: 0.018633540372670808
    RMSE: 0.136504726557987
    R^2 (explained variance): 0.9999913733609386
    """
    
    """ on the validation set """ 
    X_val = vectorize(model.val_images)
    X_val_proj = pca.transform(X_val)
    y_val_predict = neigh.predict(X_val_proj)
    y_val_predict = y_val_predict.astype(np.int8)
    
    print("Evaluation on the Validation Set")
    scoring(y_val, y_val_predict)
        
    """ RESULTS on Validation Set
    Mean Squared Error: 0.013457556935817806
    RMSE: 0.11600671073613718
    R^2 (explained variance): 0.9999937696495668

    """
    
    
    
    
    
    # Machine Learning + Computer Vision Approach
    # 5-PCA + Red Percentage + other CV features
    X_train_proj.shape
    
    red_percentages_ = []
    red_threshold = 200
    for i in range(model.train_images.shape[0]):
        red = compute_red_percentage(model.train_images[i],red_threshold)
        red_percentages_.append(red)
    red_percentages_ = np.array(red_percentages_).reshape((-1,1))
    X_train_proj = np.hstack([X_train_proj, red_percentages_])
    
    
    from sklearn.preprocessing import normalize
    X_train_proj = normalize(X_train_proj)
    
    


    """
    Full K-NN
    

    neigh = KNeighborsRegressor(n_neighbors=3)
    neigh.fit(X_train, y_train) 
    
    X_test = vectorize(model.test_images)
    y_test_predict = neigh.predict(X_test)
    y_test_predict = y_test_predict.astype(np.int8)
    
    # SCORE - RMSE
    RMSE = np.sqrt(((y_test - y_test_predict)**2).mean())
    MSE = ((y_test - y_test_predict)**2).mean()
    print("Mean Squared Error: {0}".format(MSE))
    print("RMSE: {0}".format(RMSE))
    # SCORE - R^2
    R2 = 1 - ((y_test - y_test_predict)**2).sum()/\
          ((y_test - y_test.mean()) ** 2).sum()
    print("R^2 (explained variance): {0}".format(R2))
    

    
    # RESULTS on Test Set
    Mean Squared Error: 0.031055900621118012
    RMSE: 0.17622684421256035
    R^2 (explained variance): 0.999985622268231
    """




