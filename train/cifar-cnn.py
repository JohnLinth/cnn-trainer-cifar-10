import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD

# change encoding to UTF-8 (keras didnt work with cp1252 encoding)
os.system('chcp 65001')
sys.stdout.reconfigure(encoding='utf-8')

# CIFAR-10 class names
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# unpickle function for loading CIFAR-10 data batches from https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# loads all batches of the CIFAR-10 dataset
def load_cifar10_data():
    data_batches = []
    labels_batches = []
    
    for i in range(1, 6):
        batch = unpickle(f'./data/data_batch_{i}')
        data_batches.append(batch[b'data'])
        labels_batches.append(batch[b'labels'])
    
    x_train = np.concatenate(data_batches)
    y_train = np.concatenate(labels_batches)
    
    test_batch = unpickle('./data/test_batch')
    x_test = test_batch[b'data']
    y_test = np.array(test_batch[b'labels'])
    
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_cifar10_data()

# reshape the training and testing data to 32x32x3 images and transpose the data in tensorflow format (NHWC)
x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

# normalize the numerical data to the range [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# data augmentation to add some random rotations, shifts etc. to the data (improved model acc by 1-2%)
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(x_train)

# convolutional neural network model 
model = Sequential()

# convlutional, max pooling, and batch normalization layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# flatten the 3D output to 1D for the dense layers
model.add(Flatten())

# add dense layers with dropout for regularization
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5)) # 50% dropout to avoid overfitting

# output layer (softmax for classification)
model.add(Dense(10, activation='softmax'))

# compile the model with SGD optimizer and momentum (better convergence than adam)
optimizer = SGD(learning_rate=0.001, momentum=0.9)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# set early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# train the model (30 epochs)
history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    validation_data=(x_test, y_test), epochs=30,
                    callbacks=[early_stopping])

# evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# plot accuracy
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# plot validation loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# export the model for use
model.save('cifar10_cnn_model.h5')
