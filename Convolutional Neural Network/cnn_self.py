# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 14:27:02 2019

@author: nsingh1
"""

# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialise CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation = 'relu'))

# Step 2 - Pooling (Reducing size of feature detector/map to get features and reduce computation)
classifier.add(MaxPooling2D(pool_size=(2,2)))


### Adding Another Convolutional an pooling layer to increase the accuracy of test set
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


# Step 3 - Flattening ( Combing All Pooling Feature map into 1 single vector)
classifier.add(Flatten())

# Step 4 - Adding Fully connected layer and Output layer
classifier.add(Dense(output_dim = 128, #How many nodes in the hidden layer 
                     activation= 'relu'))
classifier.add(Dense(output_dim = 1, 
                     activation= 'sigmoid'))# as it is bianry classifaction, if it has more than 2 output the use softmax function

# Step 5 - Compiling CNN
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])



##  Part 2 - Fitting CNN to Images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

testing_set = test_datagen.flow_from_directory('dataset/test_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=testing_set,
                        validation_steps=2000)

## Part 3 - Making Single prediction

from keras.preprocessing import image
import numpy as np

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', 
                            target_size=(64, 64)) # Dog image
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
prediction = classifier.predict(test_image)
print(training_set.class_indices)

if int(prediction[0][0])==1:
    print("Predicted Value is Dog")
    
else:
    print("Predicted Value is Cat")
    


