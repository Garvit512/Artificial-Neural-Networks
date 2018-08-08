# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 11:25:36 2018

@author: garvi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_X = LabelEncoder()
X[:,1] = LabelEncoder_X.fit_transform(X[:,1])
X[:,2] = LabelEncoder_X.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features= [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Import libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initilizing the ANN
classifier = Sequential()

# Adding I/P layer and first hidden Layer
classifier.add(Dense(output_dim=6,activation='relu', kernel_initializer= 'uniform', input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, activation='relu', kernel_initializer='uniform'))

# Adding O/P Layer
classifier.add(Dense(output_dim = 1, activation='sigmoid', kernel_initializer='uniform'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)