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
#from skimage.feature import match_template


# Tool functions
def compute_red_percentage(img, threshold):
    return len(img[img>=threshold])/(img.shape[0]*img.shape[1])
   
def vectorize_single(image, keep_edge):
    """ Convert the images into feature vectors of size (width x height)"""
    temp_gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    temp_gray_img = gaussian_filter(temp_gray_img, sigma = 2)
    
    if keep_edge: # we conserve only the edges in the image
        temp_gray_img = cv2.Canny(np.uint8(temp_gray_img*255),30,140) 
    
    temp_gray_img = temp_gray_img.ravel()
    return temp_gray_img

def vectorize(images, keep_edge):
    """ Convert the images into feature vectors of size (width x height)"""
    X_vec = np.zeros((images.shape[0], images.shape[1]*images.shape[2]))
    for i in range(images.shape[0]):
        X_vec[i] = vectorize_single(images[i], keep_edge)
    return X_vec


class StopSigns:
    
    # Attributes of the data
    height, width, channels = 275, 275, 3
    min_heading = -80
    max_heading = 80
    num_image_per_heading = 30
    
    # Ratio of the training, validation and testing set
    p_train = 0.6
    p_val = 0.2
    p_test= 0.2
    
    # Parameters to tune
    #red_threshold = 200

    
    
    def __init__(self):
        """
        Attributes:
            headings: all the possible range
            
            train_images, val_images, test_images: images in a (height, width, channel shapes)
                regrouped in a 4-d array (used for the computer vision features detection)
            train_labels, val_labels, test_labels: their corresponding headings
            
            X_train, X_val, X_test: the vectorized images (one row correspond to one image)
            X_train_proj, X_val_proj, X_test_proj: their projection on the first principal vectors (PCA)
            y_train, y_val, y_test: their corresponding labels (same as train_labels, ...)
            
                
            
        """
        self.headings = np.array(range(self.min_heading,self.max_heading+1))
        self.train_images, self.train_labels = None, None
        self.val_images, self.val_labels = None, None
        self.test_images, self.test_labels = None, None
        self.X_train, self.X_train_proj, self.y_train = None, None, None
        self.X_val, self.X_val_proj, self.y_val = None, None, None
        self.X_test, self.X_test_proj, self.y_test = None, None, None
        self.y_train_pred, self.y_val_pred, self.y_test_pred = None, None, None
        
        
        """
        Fitting parameters
        """
        #self.w_red_percentage = [] NOT USED
        self.pca = None
        self.neigh = None
        
        
    
    def split_data(self, p_train = p_train, p_val = p_val, p_test =  p_test):
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
    

    def vectorize_sets(self, keep_edge = 0):
        """
        This function is preprocessing the data and store them in a matrix where
        each row represents a 275x275 image.
        The preprocessing step includes:
            covert the image in grayscale
            keep the edge of the image only (keep_edge parameter on/off)
            
        """
        print("Vectorizing data")
        self.X_train = vectorize(self.train_images, keep_edge)
        self.X_val = vectorize(self.val_images, keep_edge)
        self.X_test = vectorize(self.test_images, keep_edge)
        self.y_train = self.train_labels
        self.y_val = self.val_labels
        self.y_test = self.test_labels
        
        
        
        
    def do_pca(self, k ,plot = 0):
        print("Principal Component Analysis")
        self.pca = PCA(n_components=k)
        self.pca.fit(self.X_train)
        print("\t explained variance ratio_: ",self.pca.explained_variance_ratio_) 
        print("\t total explained variance ratio",self.pca.explained_variance_ratio_.sum())
        if plot:
            plt.title("explained variance ratio function of the i_th component")
            plt.plot(self.pca.explained_variance_ratio_)
            plt.show()
            
            
    def pca_transform(self, plot = 0):
        print("Computing the Principal Components")
        self.X_train_proj = self.pca.transform(self.X_train)
        self.X_val_proj = self.pca.transform(self.X_val)
        self.X_test_proj = self.pca.transform(self.X_test)
        
        if plot:
            num_training = int(self.num_image_per_heading*self.p_train)
            n = self.max_heading-self.min_heading+1
            for i in range(n):
                plt.scatter(self.X_train_proj[:,0][num_training*i:num_training*(i+1)]\
                            , self.X_train_proj[:,1][num_training*i:num_training*(i+1)]\
                            , c=str(i/n))
            plt.title("2D projection of the images after PCA (grayscale represents the heading)")
            plt.show()
            
            
    def fit(self, n_neighbors, verbose = 1):
        if verbose:
            print("Fitting the training data with k-nearest neighbors")
        self.neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.neigh.fit(self.X_train_proj, self.y_train) 
        
        
    def predict(self):
        self.y_train_pred = self.neigh.predict(self.X_train_proj).astype(np.int8)
        self.y_val_pred = self.neigh.predict(self.X_val_proj).astype(np.int8)
        self.y_test_pred = self.neigh.predict(self.X_test_proj).astype(np.int8)
        
      
    def score(self, mode):
        if mode == "training":
            y_true = self.y_train
            y_pred = self.y_train_pred
        elif mode == "validation":
            y_true = self.y_val
            y_pred = self.y_val_pred
        elif mode == "test":
            y_true = self.y_test
            y_pred = self.y_test_pred
        else: 
            raise NameError("invalid mode parameter. training, validation or test accepted")
        print ("Evaluation on the {} set:".format(mode))
        RMSE = np.sqrt(((y_true - y_pred)**2).mean())
        MSE = ((y_true - y_pred)**2).mean()
        print("\t Mean Squared Error: {0}".format(MSE))
        print("\t RMSE: {0}".format(RMSE))
        # SCORE - R^2
        R2 = 1 - ((y_true - y_pred)**2).sum()/\
              ((y_true - y_true.mean()) ** 2).sum()
        print("\t R^2: {0}".format(R2)) 
            
        
    def accuracy(self, mode):
        if mode == "training":
            y_true = self.y_train
            y_pred = self.y_train_pred
        elif mode == "validation":
            y_true = self.y_val
            y_pred = self.y_val_pred
        elif mode == "test":
            y_true = self.y_test
            y_pred = self.y_test_pred
        else: 
            raise NameError("invalid mode parameter. training, validation or test accepted")
        classif = y_true - y_pred
        acc = classif[classif==0].shape[0]/y_true.shape[0]
        print ("Accuracy of the {0} set: {1}".format(mode, acc))
        return classif[classif==0].shape[0]
        
            
    


    
    def fit_red(self, X, y, plot = 0):        
        
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
    
    # Loading, splitting and preprocessing the data
    model = StopSigns()
    model.split_data()
    model.vectorize_sets(keep_edge=0)
    
    # PCA Looking at the explained variance by the principal components and choosing k
    model.do_pca(10, plot = 1)

    nb_principal_components = 5
    model.do_pca(nb_principal_components)
    
    # Project the data + 2D visualization
    model.pca_transform(plot = 1)
    
    # Fitting the model with k-Nearest Neighbor
    n_neighbors = 3
    model.fit(n_neighbors)
    
    # Prediction
    model.predict()

    # Performance evaluation
    model.score("training")
    model.score("validation")
    model.score("test")
    
    model.accuracy("training")
    model.accuracy("validation")
    model.accuracy("test")





    """ For a new image """
    print(" #### Prediction on 'real' images ###")
    headings_new_img=[]
    # loading the new images
    real_img = mpimg.imread("real_images/1.jpg")
    real_img2 = mpimg.imread("real_images/2.jpg")
    
    # evaluating the previous model on the first image
    real_img_vec = vectorize_single(real_img, keep_edge=1)
    real_img_proj = model.pca.transform(real_img_vec.reshape((1,-1)))
    for n_neighbors in range(1,500):
        model.fit(n_neighbors, verbose=0)
        headings_new_img.append(model.neigh.predict(real_img_proj)[0][0])
    plt.title("Heading function of the number of nearest neighbors")
    plt.plot(headings_new_img)
    plt.show()
    
    
    
    
    
    """ Computer Vision Features """
    
    # RED PERCENTAGE 
    red_threshold = 200
    
    # Red pixels percentage distribution function of the heading
    red_distribution = []
    for i in range(model.train_images.shape[0]):
        red = compute_red_percentage(model.train_images[i][:,:,0],red_threshold)
        red_distribution.append(red)
    red_distribution = np.array(red_distribution).reshape((-1,1))
    
    plt.title("Red pixel percentage function of heading")
    plt.plot(model.headings,red_distribution[::18],c='r')
    plt.show()
    
    
    
    # Compute red percentage on real images
    temp_real_img = real_img.copy()
    plt.imshow(temp_real_img)
    plt.show()

    
    only_red = np.zeros((275,275), dtype=np.uint16)
    cb = 0
    cg = 0
    alpha = 0.04
    for i in range(275):
        for j in range(275):
            if (temp_real_img[i,j,1]==0):
                cb = temp_real_img[i,j,0]
            else: 
                cb = temp_real_img[i,j,0]/temp_real_img[i,j,1]
            if (temp_real_img[i,j,2]==0):
                cg = temp_real_img[i,j,0]
            else: 
                cg = temp_real_img[i,j,0]/temp_real_img[i,j,2]
            R = cb*cg - alpha*(cb+cg)**2
            only_red[i,j] = R
            
    plt.hist(only_red.ravel(), bins=40)
    plt.show()
    
    # threshold
    only_red[only_red<12]=0
    only_red[only_red>=12]=1

    plt.imshow(only_red,cmap='gray')
    plt.axis("off")
    plt.show()
    
    
    print("The red percentage of the real image is {0}".format(\
          len(only_red[only_red==1])/(only_red.shape[0]*only_red.shape[1])))
    
    
    
    
    
    



