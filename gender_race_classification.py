# -*- coding: utf-8 -*-
"""
@author: Alex Zhao
"""

# import necessary packages
import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import pandas as pd
import seaborn as sn

from keras.engine import Model
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, Activation, Dropout, Flatten, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import img_to_array

from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split


def build_model(model_type, sample_size_per_group, split, epoch, num_of_batches, epoch_patience):
  
    # extract data from image
    print('Extracting data from images...')
    
    X = []
    y_race = []
    y_gender = []
    
    image_files = []
    image_sample = []
    
    race = ['asian', 'black', 'indian', 'latino', 'white']
    gender = ['man', 'woman']
    
    image_files = os.listdir('images/asianman')
    image_sample = random.sample(image_files, sample_size_per_group)
    for filename in image_sample:
        if 'jpg' in filename:
            image = cv2.imread('images/asianman/' + filename)
            image = cv2.resize(image, (224,224))
            image = img_to_array(image)
            X.append(image)
            y_gender.append('man')
            y_race.append('asian')
    
    image_files = os.listdir('images/asianwoman')
    image_sample = random.sample(image_files, sample_size_per_group)
    for filename in image_sample:
        if 'jpg' in filename:
            image = cv2.imread('images/asianwoman/' + filename)
            image = cv2.resize(image, (224,224))
            image = img_to_array(image)
            X.append(image)
            y_gender.append('woman')
            y_race.append('asian')
    
    image_files = os.listdir('images/blackman')
    image_sample = random.sample(image_files, sample_size_per_group)
    for filename in image_sample:
        if 'jpg' in filename:
            image = cv2.imread('images/blackman/' + filename)
            image = cv2.resize(image, (224,224))
            image = img_to_array(image)
            X.append(image)
            y_gender.append('man')
            y_race.append('black')
    
    image_files = os.listdir('images/blackwoman')
    image_sample = random.sample(image_files, sample_size_per_group)
    for filename in image_sample:
        if 'jpg' in filename:
            image = cv2.imread('images/blackwoman/' + filename)
            image = cv2.resize(image, (224,224))
            image = img_to_array(image)
            X.append(image)
            y_gender.append('woman')
            y_race.append('black')
            
    image_files = os.listdir('images/indianman')
    image_sample = random.sample(image_files, sample_size_per_group)
    for filename in image_sample:
        if 'jpg' in filename:
            image = cv2.imread('images/indianman/' + filename)
            image = cv2.resize(image, (224,224))
            image = img_to_array(image)
            X.append(image)
            y_gender.append('man')
            y_race.append('indian')
                
    image_files = os.listdir('images/indianwoman')
    image_sample = random.sample(image_files, sample_size_per_group)
    for filename in image_sample:
        if 'jpg' in filename:
            image = cv2.imread('images/indianwoman/' + filename)
            image = cv2.resize(image, (224,224))
            image = img_to_array(image)
            X.append(image)
            y_gender.append('woman')
            y_race.append('indian')
          
    image_files = os.listdir('images/latinoman')
    image_sample = random.sample(image_files, sample_size_per_group)
    for filename in image_sample:
        if 'jpg' in filename:
            image = cv2.imread('images/latinoman/' + filename)
            image = cv2.resize(image, (224,224))
            image = img_to_array(image)
            X.append(image)
            y_gender.append('man')
            y_race.append('latino')
    
    image_files = os.listdir('images/latinowoman')
    image_sample = random.sample(image_files, sample_size_per_group)
    for filename in image_sample:
        if 'jpg' in filename:
            image = cv2.imread('images/latinowoman/' + filename)
            image = cv2.resize(image, (224,224))
            image = img_to_array(image) 
            X.append(image)
            y_gender.append('woman')
            y_race.append('latino')
         
    image_files = os.listdir('images/whiteman')
    image_sample = random.sample(image_files, sample_size_per_group)
    for filename in image_sample:
        if 'jpg' in filename:
            image = cv2.imread('images/whiteman/' + filename)
            image = cv2.resize(image, (224,224))
            image = img_to_array(image)     
            X.append(image)
            y_gender.append('man')
            y_race.append('white')
          
    image_files = os.listdir('images/whitewoman')
    image_sample = random.sample(image_files, sample_size_per_group)
    for filename in image_sample:
        if 'jpg' in filename:
            image = cv2.imread('images/whitewoman/' + filename)
            image = cv2.resize(image, (224,224))
            image = img_to_array(image)
            X.append(image)
            y_gender.append('woman')
            y_race.append('white')
    
    # prepare the data
    print('Processing data...')
    
    X = np.array(X, dtype="float") / 255.0
    y_gender = np.array(y_gender)
    y_race = np.array(y_race)
    
    enc_race = preprocessing.LabelEncoder()
    enc_gender = preprocessing.LabelEncoder()
    
    enc_race.fit(race)
    enc_gender.fit(gender)
    
    y_race = enc_race.transform(y_race)
    y_gender = enc_gender.transform(y_gender)
    
    y_race = to_categorical(y_race, num_classes = len(race))
    y_gender = to_categorical(y_gender, num_classes = len(gender))
    
    X_gender_train, X_gender_test, y_gender_train, y_gender_test = train_test_split(X, y_gender, test_size = 0.25, random_state = 42)
    
    X_race_train, X_race_test, y_race_train, y_race_test = train_test_split(X, y_race, test_size = 0.25, random_state = 42)
    
    # build model
    print('Loading face model...')
    
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
     
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
     
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
     
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
     
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
     
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    
    model.load_weights('vgg_face_weights.h5')
    
    kfold = KFold(n_splits = split, shuffle = True)
    early_stop = EarlyStopping(monitor = 'loss', patience = epoch_patience)
    
    for layer in model.layers[:-7]:
        layer.trainable = False
    
    # gender model
    if  model_type == 'gender':
        print('Building gender model...')
        i = 0
        gender_model = [None] * split
        gender_model_train = [None] * split
        gender_model_evaluate = [None] * split
        
        for train, test in kfold.split(X_gender_train, y_gender_train):
            print('Training model ' + str(i) + '...')
            gender_model_output = Sequential()
            gender_model_output = Convolution2D(len(gender), (1, 1), name = 'predictions')(model.layers[-4].output)
            gender_model_output = Flatten()(gender_model_output)
            gender_model_output = Activation('softmax')(gender_model_output)
            
            gender_model[i] = Model(inputs = model.input, outputs = gender_model_output)
            
            gender_model[i].compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])
            
            gender_model_train[i] = gender_model[i].fit(X[train], y_gender[train],
                                                  validation_data = (X_gender_train[test], y_gender_train[test]),
                                                  batch_size = len(X_gender_train[train])//num_of_batches,
                                                  epochs = epoch, callbacks = [early_stop], verbose = 1)
            
            print('Evaluating model ' + str(i) + '...')
            gender_model_evaluate[i] = gender_model[i].evaluate(X_gender_test, y_gender_test, batch_size = len(X_gender_train[train])//num_of_batches)
            
            i += 1
        
        best_model = 0
        for m in range (len(gender_model_evaluate) - 1):
            if gender_model_evaluate[m][1] > gender_model_evaluate[m + 1][1]:
                best_model = m
             
        gender_model[m].save('gender_model.h5')
        gender_model[m].save_weights("gender_model_weights.h5")
        
        plt.title('Gender Model Loss')
        plt.plot(gender_model_train[0].history['loss'], label = 'train')
        plt.plot(gender_model_train[0].history['val_loss'], label = 'val')
        plt.legend(loc='upper right')
        plt.savefig('gender_model_loss.png')
        plt.show()
        
        plt.title('Gender Model Accuracy')
        plt.plot(gender_model_train[0].history['accuracy'], label = 'train')
        plt.plot(gender_model_train[0].history['val_accuracy'], label = 'val')
        plt.legend(loc='upper right')
        plt.savefig('gender_model_accuracy.png')
        plt.show()
        
        np.savetxt('gender_model_evaluation.txt', gender_model_evaluate[0])
        print ('Test Loss, Test Accuracy:', gender_model_evaluate[0])
        
        gender_predicted = gender_model[0].predict(X_gender_test)
        gender_predicted_class = []
        gender_actual_class = []
        
        for p in range (0, len(gender_predicted)):
            gender_predicted_class.append(gender[np.argmax(gender_predicted[p])])
            gender_actual_class.append(gender[np.argmax(y_gender_test[p])])
        
        cm = confusion_matrix(gender_actual_class, gender_predicted_class, gender)
        
        df_cm = pd.DataFrame(cm, index = gender, columns = gender)
        sn.heatmap(df_cm, annot=True)
        plt.title('Gender Confusion Matrix')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.savefig('gender_model_cm.png')
        plt.show()
        
    # race model
    if  model_type == 'race':
        print('Building race model...')
        i = 0
        race_model = [None] * split
        race_model_train = [None] * split
        race_model_evaluate = [None] * split
        
        for train, test in kfold.split(X_race_train, y_race_train):
            print('Training model ' + str(i) + '...')
            race_model_output = Sequential()
            race_model_output = Convolution2D(len(race), (1, 1), name = 'predictions')(model.layers[-4].output)
            race_model_output = Flatten()(race_model_output)
            race_model_output = Activation('softmax')(race_model_output)
            
            race_model[i] = Model(inputs = model.input, outputs = race_model_output)
            
            race_model[i].compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])
            
            race_model_train[i] = race_model[i].fit(X_race_train[train], y_race_train[train],
                                              validation_data = (X_race_train[test], y_race_train[test]),
                                              batch_size = len(X_race_train[train])//num_of_batches,
                                              epochs = epoch, callbacks = [early_stop], verbose = 1)
            
            print('Evaluating model ' + str(i) + '...')
            race_model_evaluate[i] = race_model[i].evaluate(X_race_test, y_race_test, batch_size = len(X_race_train[train])//num_of_batches)
            
            i += 1
        
        best_model = 0
        for m in range (len(race_model_evaluate) - 1):
            if race_model_evaluate[m][1] > race_model_evaluate[m + 1][1]:
                best_model = m
        
        race_model[m].save('race_model.h5')
        race_model[m].save_weights("race_model_weights.h5")
        
        plt.title('Race Model Loss')
        plt.plot(race_model_train[best_model].history['loss'], label = 'train')
        plt.plot(race_model_train[best_model].history['val_loss'], label = 'val')
        plt.legend(loc='upper right')
        plt.savefig('race_model_loss.png')
        plt.show()

        plt.title('Race Model Accuracy')
        plt.plot(race_model_train[best_model].history['accuracy'], label = 'train')
        plt.plot(race_model_train[best_model].history['val_accuracy'], label = 'val')
        plt.legend(loc='upper right')
        plt.savefig('race_model_accuracy.png')
        plt.show()

        np.savetxt('race_model_evaluation.txt', race_model_evaluate[0])
        print ('Test Loss, Test Accuracy:', race_model_evaluate[best_model])
        
        race_predicted = race_model[best_model].predict(X_race_test)
        race_predicted_class = []
        race_actual_class = []
        
        for p in range (len(race_predicted)):
            race_predicted_class.append(race[np.argmax(race_predicted[p])])
            race_actual_class.append(race[np.argmax(y_race_test[p])])
        
        cm = confusion_matrix(race_actual_class, race_predicted_class, race)
        
        df_cm = pd.DataFrame(cm, index = race, columns = race)
        sn.heatmap(df_cm, annot=True)
        plt.title('Race Confusion Matrix')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.savefig('race_model_cm.png')
        plt.show()