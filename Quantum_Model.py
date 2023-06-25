####################################imort libraries############################################
import pennylane as qml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
import cv2
import imutils
import keras 
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model, load_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import classification_report

####################################data preprocessing############################################
# Path to the data directory
data_dir = 'E:\QCNN\Brain Tumor Data Set\Brain Tumor Data Set'
# Get the list of all the images
images = os.listdir(data_dir)
# Get the list of all the images
data = []
labels = []
for i in ['Healthy', 'Brain Tumor']:
    path = os.path.join(data_dir,i)
    for img in os.listdir(path):
        try:
            image = cv2.imread(os.path.join(path,img))
            print(os.path.join(path,img))
            image = cv2.resize(image, (224,224))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")
# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)
# Shuffle the data
data,labels = shuffle(data,labels, random_state=42)
# Split the data into train and test set
train_data,test_data,train_labels,test_labels = train_test_split(data,labels,test_size=0.1,random_state=42)
# Normalize the data
train_data = train_data / 255.0
test_data = test_data / 255.0
# Onehot encoding the labels
train_labels = pd.get_dummies(train_labels).values
test_labels = pd.get_dummies(test_labels).values
# Data Augmentation
train_datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode="nearest")
train_datagen.fit(train_data)

####################################Quantum Model############################################
# Quantum circuit
def circuit(params, wires):
    qml.RX(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.RZ(params[2], wires=wires[2])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.CNOT(wires=[wires[2], wires[0]])
    qml.RX(params[3], wires=wires[0])
    qml.RY(params[4], wires=wires[1])
    qml.RZ(params[5], wires=wires[2])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.CNOT(wires=[wires[2], wires[0]])
    qml.RX(params[6], wires=wires[0])
    qml.RY(params[7], wires=wires[1])
    qml.RZ(params[8], wires=wires[2])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.CNOT(wires=[wires[2], wires[0]])
    qml.RX(params[9], wires=wires[0])
    qml.RY(params[10], wires=wires[1])
    qml.RZ(params[11], wires=wires[2])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.CNOT(wires=[wires[2], wires[0]])
    qml.RX(params[12], wires=wires[0])
    qml.RY(params[13], wires=wires[1])
    qml.RZ(params[14], wires=wires[2])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.CNOT(wires=[wires[2], wires[0]])
    qml.RX(params[15], wires=wires[0])
    qml.RY(params[16], wires=wires[1])
    qml.RZ(params[17], wires=wires[2])  
    return qml.expval(qml.PauliZ(0))

#params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
#drawer = qml.draw(circuit)
#print(drawer(params, wires=range(3)))

# Quantum device
dev = qml.device("default.qubit", wires=3)

# Quantum node
@qml.qnode(dev)
def quantum_model(params, wires):
    return circuit(params, wires)

# Cost function
def cost(var, features, labels):
    predictions = [quantum_model(var, wires=range(3)) for feature in features]
    return square_loss(labels, predictions)

# Square loss
def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss

# Gradient descent optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.4)

# Training the quantum model
def training(var, features, labels, steps, shots):
    for i in range(steps):
        var = opt.step(lambda v: cost(v, features, labels), var)
        if (i + 1) % 5 == 0:
            preds = [np.sign(quantum_model(var, wires=range(3))) for feature in features]
            acc = accuracy(labels, preds)
            loss = cost(var, features, labels)
            print("Step {}: Accuracy = {}, Loss = {}".format(i + 1, acc, loss))
    return var

# Accuracy
def accuracy(labels, predictions):
    """
    Computes the accuracy of the predictions.

    Args:
        labels (array): True labels.
        predictions (array): Predicted labels.

    Returns:
        float: The accuracy of the predictions.
    """
    predictions = np.array(predictions)
    predictions = predictions.reshape(-1, 1)
    return np.mean(np.allclose(labels, predictions))

# Initial parameters
np.random.seed(0)
var_init = np.random.randn(18)

# Training the quantum model
var = training(var_init, train_data, train_labels, steps=100, shots=1000)

# Testing the quantum model
preds = [np.sign(quantum_model(var, wires=range(3))) for feature in test_data]
acc = accuracy(test_labels, preds)
loss = cost(var, test_data, test_labels)
print("Accuracy on test data = {}%".format(acc * 100))
print("Loss on test data = {}".format(loss))

####################################Classical Model############################################
# Building the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
history = model.fit(train_datagen.flow(train_data, train_labels, batch_size=32), epochs=10, validation_data=(test_data, test_labels))

# Evaluating the model
model.evaluate(test_data, test_labels)

# Predicting the model
preds = model.predict(test_data)
preds = np.argmax(preds, axis=1)
test_labels = np.argmax(test_labels, axis=1)

# Classification report
print(classification_report(test_labels, preds, target_names=['Healthy', 'Brain Tumor']))

####################################Comparing the models############################################
# Plotting the accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting the loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

####################################Saving the models############################################
# Saving the quantum model
qml.save(var, 'E:\QCNN\Brain Tumor Data Set\Brain Tumor Data Set\Quantum Model')

# Saving the classical model
model.save('E:\QCNN\Brain Tumor Data Set\Brain Tumor Data Set\Classical Model')

####################################Loading the models############################################
# Loading the quantum model
var = qml.load('E:\QCNN\Brain Tumor Data Set\Brain Tumor Data Set\Quantum Model')

# Loading the classical model
model = load_model('E:\QCNN\Brain Tumor Data Set\Brain Tumor Data Set\Classical Model')

####################################Testing the models############################################
# Testing the quantum model
preds = [np.sign(quantum_model(var, wires=range(3))) for feature in test_data]
acc = accuracy(test_labels, preds)
loss = cost(var, test_data, test_labels)
print("Accuracy on test data = {}%".format(acc * 100))
print("Loss on test data = {}".format(loss))

# Testing the classical model
model.evaluate(test_data, test_labels)

####################################Testing the models on new data############################################
# Path to the data directory
data_dir = 'E:\QCNN\Brain Tumor Data Set\Brain Tumor Data Set'
# Get the list of all the images
images = os.listdir(data_dir)
# Get the list of all the images
data = []
labels = []
for i in ['Healthy', 'Brain Tumor']:
    path = os.path.join(data_dir,i)
    for img in os.listdir(path):
        try:
            image = cv2.imread(os.path.join(path,img))
            print(os.path.join(path,img))
            image = cv2.resize(image, (224,224))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")
# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)
# Shuffle the data
data,labels = shuffle(data,labels, random_state=42)
# Split the data into train and test set
train_data,test_data,train_labels,test_labels = train_test_split(data,labels,test_size=0.1,random_state=42)
# Normalize the data
train_data = train_data / 255.0
test_data = test_data / 255.0
# Onehot encoding the labels
train_labels = pd.get_dummies(train_labels).values
test_labels = pd.get_dummies(test_labels).values
# Data Augmentation
train_datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode="nearest")
train_datagen.fit(train_data)

# Testing the quantum model
preds = [np.sign(quantum_model(var, wires=range(3))) for feature in test_data]
acc = accuracy(test_labels, preds)
loss = cost(var, test_data, test_labels)
print("Accuracy on test data = {}%".format(acc * 100))
print("Loss on test data = {}".format(loss))

# Testing the classical model
model.evaluate(test_data, test_labels)




