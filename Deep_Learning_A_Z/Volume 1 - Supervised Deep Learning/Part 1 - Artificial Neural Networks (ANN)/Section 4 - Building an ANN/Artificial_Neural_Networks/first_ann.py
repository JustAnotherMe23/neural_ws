# This is a practice Artificial Neural Network
# The problem being solved is based off of a model bank with fake data
# The bank has customers that have left for whatever reason
# The goal is to find why these customers have left using information such as account balance and gender
# The last column of the data states whether or not the customer has left the bank


import numpy as np #Math operations library
import matplotlib.pyplot as plt #Visualization library
import pandas as pd #Matrix handler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler #Methods to change categorical strings to numbers and scaling ability
from sklearn.model_selection import train_test_split
import keras # Brings in tensorflow with it
from keras.models import Sequential # Used for initialization of ANN
from keras.layers import Dense # adds layers to ANN

# Gather data
dataset = pd.read_csv('Churn_Modelling.csv') #Import data from csv
x = dataset.iloc[:, 3:13].values #Pulls features from data
y = dataset.iloc[:, 13].values # Pulls result from data

# Preprocessing
label_encoder_x_1 = LabelEncoder()
label_encoder_x_2 = LabelEncoder() # For text categories, gives int value instead
x[:, 1] = label_encoder_x_1.fit_transform(x[:, 1])
x[:, 2] = label_encoder_x_2.fit_transform(x[:, 2])


onehotencoder = OneHotEncoder(categorical_features=[1]) # Separates categorical data into binary columns
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:] # Removes first column to prevent dummy variable trap

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) # Creates train and test sets

sc = StandardScaler() # Scales data to make initial weight value close to end value
x_train = sc.fit_transform(x_train) # Computes mean and applies transformation
x_test = sc.transform(x_test) # Uses the mean from the above method

# Notes
# Categorical data is no longer from 0 to 1
# Scaling does not apply from 0 to 1
# Categories of 1 have the same value, as do categories of 0

classifier = Sequential() # This is the ANN object
classifier.add(Dense(input_dim=11, units=6, kernel_initializer='uniform', activation='relu')) #Creates first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu')) # Second layer. Input dim is known from previous layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid')) # Output layer. Only 1 ouput category, sigmoid activation to get probability of sureness

# Note: Softmax applies to a dependent variable that has more than 2 categories
# i.e. fMRI categorizations

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
# Notes
# adam is a kind of stochastic gradient descent
# For multivariabel, use categorical cross entropy
# Accuracy is predefined and shows accuracy

classifier.fit(x_train, y_train, batch_size=10, epochs=100)