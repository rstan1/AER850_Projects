"""
Created on Wed Oct 30 20:05:41 2024
Project 2
@author: robert_stangaciu_501095883
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Step 1: Define input image shape
IMG_HEIGHT, IMG_WIDTH, CHANNELS = 500, 500, 3
BATCH_SIZE = 32

# Step 2: Set up data directories
train_dir = '/Users/robstan/Desktop/AER 850/Github/AER850_Projects/Data/Train'
validation_dir = '/Users/robstan/Desktop/AER 850/Github/AER850_Projects/Data/valid'

# Step 3: Data augmentation for training and rescaling for validation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Step 4: Create training and validation generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Step 2: Define the CNN architecture
model = Sequential()

# Convolutional and MaxPooling layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer
model.add(Flatten())

# Fully connected Dense layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization

model.add(Dense(3, activation='softmax'))  # 3 neurons for 3 classes

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display the model architecture
model.summary()
