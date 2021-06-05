# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:46:18 2020

@author: Xieyuan Huang U3190995
This is for handwriting recognition by CNN model

!!IMPORTANT: parameter tuning will take about 35 hours running with cpu intel i7 9750H

"""
#import libraries for loading MNIST dataset, data visualization, constructing CNN and evaluation method
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras import backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
# hyperparameter optimization
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from time import time

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

#Visualize 20 samples in x_train
num_row = 4
num_col = 5
samples = x_train[:20]
labels = y_train[:20]
# plot images using for loop
fig, axes = plt.subplots(num_row, num_col)
for i in range(20):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(samples[i], cmap='gray')
    ax.set_title('Label: {}'.format(labels[i]))
plt.tight_layout()
plt.show()

#Reshape the data into 28 by 28
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#Normalization: divide x_train and x_test with 255 since the range of each pixel is from 0 to 255
#So that all value can be from 0-1 which improve the performance of the model
#Reference: https://towardsdatascience.com/why-data-should-be-normalized-before-training-a-neural-network-c626b7f66c7d
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


num_classes=10
input_shape = (28, 28, 1)

# Create a CNN model in a function 
# we will optimize the activation function of the 2nd and 3rd convolution layers 
# Reference: https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
#https://github.com/tpvt99/character-recognition-cnn/blob/master/cnn.py
#https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
def create_cnn_model(conv_activation='tanh'):
    # create model
    model = Sequential()
    
    # Convolutional layer 1
    model.add(Conv2D(6, kernel_size=(5, 5),activation='tanh',input_shape=input_shape)) 
    # Pooling layer 1
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional layer 2
    model.add(Conv2D(16, (5, 5), activation=conv_activation))
    # Pooling layer 2
    model.add(MaxPooling2D(pool_size=(2, 2)))   
    # Convert 2D maps to single dimension vector
    model.add(Flatten())
    # Fully connected layer 1 with dropout rate 0.3
    model.add(Dense(120, activation=conv_activation))
    #model.add(Dropout(0.3))
    # Fully connected layer 2 with dropout rate 0.5
    model.add(Dense(84, activation=conv_activation))
    #model.add(Dropout(0.5))
    #Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    
    
    # Compile model
    model.compile( 
        optimizer='Adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        )    
    return model

#Display the summary of the model
cnn = create_cnn_model()
cnn.summary()
#Train the model
hist = cnn.fit(
    x_train, y_train,
    batch_size=64,
    epochs=1,
    verbose=1,
    validation_data=(x_test, y_test)
    )

#####################################Parameter tuning will take about 36 hours##############
# optimize model 
start = time()

# create model
model = KerasClassifier(build_fn=create_cnn_model, verbose=1)
# define parameters and values for grid search 
param_grid = {
    #'pool_type': ['max', 'average'],
    'conv_activation': ['relu','tanh'],  
    'batch_size': [64,96,128],
    'epochs': [10,15,20]
}

#Start searching
grs = GridSearchCV(estimator=model, param_grid=param_grid, cv=7)
grs_result = grs.fit(x_train, y_train)

# summarize results
print("Best Hyper Parameters:",grs.best_params_)

# define function to display the results of the grid search
def display_cv_results(search_results):
    print('Best score = {:.4f} using {}'.format(search_results.best_score_, search_results.best_params_))
    means = search_results.cv_results_['mean_test_score']
    stds = search_results.cv_results_['std_test_score']
    params = search_results.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('mean test accuracy +/- std = {:.4f} +/- {:.4f} with: {}'.format(mean, stdev, param))    

# summarize results
print('time for grid search = {:.0f} sec'.format(time()-start))
display_cv_results(grs_result)

#Predict based on x_test
y_pred=grs.predict(x_test)
#Display the accuracy score of the data
from sklearn import metrics
#Convert y_pred to binary matrix
y_pred = keras.utils.to_categorical(y_pred, 10)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred,axis = 1) 
# Convert validation observations to one hot vectors
y_true = np.argmax(y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#Confusion matrix reference: https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
#Plot confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 

