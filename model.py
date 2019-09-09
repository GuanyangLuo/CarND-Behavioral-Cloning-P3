import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

image_width = 320
image_height = 160

def process_CSV(directory_list):
    """
    Read the CSV file and return a list of the lines
    (Also remove the last empty line in the CSV)
    """
    lines = []
    for dir_name in directory_list:
        filename = dir_name + 'driving_log.csv'
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
        lines.pop() # remove the empty line  
    return lines


def augment_flip(image, steering_angle):
    """
    Flip the image and steering_angle and return them
    """
    image_flipped = np.fliplr(image)
    steering_angle_flipped = np.multiply(-1.0,steering_angle)
    return image_flipped, steering_angle_flipped


def process_line(line):
    """
    Process a line from driving_log CSV. Return the center, left, 
    and right images and coresponding steering angles, as well as
    the flipped version of these data
    """
    # Steering angle adjustment for left and right images
    correction = 0.2
    # Get the images from the line
    images = []
    for i in range(3):
        image = ndimage.imread(line[i])
        images.append(image)
    # Generate steering_angles for each image
    steering_angle = float(line[3])
    steering_angles = [steering_angle, steering_angle+correction, steering_angle-correction]
    # Augment the data
    images_flipped, steering_angles_flipped = augment_flip(images,steering_angles)
    images.extend(images_flipped)
    steering_angles.extend(steering_angles_flipped)
    return images, steering_angles
    

def generator(samples, batch_size=32):
    """
    Generate images and steering angles from CSV lines in batches
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        # Batching
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            # Process a batch of data
            images = []
            angles = []
            for batch_sample in batch_samples:
                batch_images, batch_angles = process_line(batch_sample)
                images.extend(batch_images)
                angles.extend(batch_angles)

            # Convert to numpy array
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
    
# Read CSV
CSV_directories = ['./my_data/', './track1_reverse/', './track2/', './track2_reverse/']
lines = process_CSV(CSV_directories)
# For the first dataset, I started at a turn but didn't start driving 
# until sometime after pressing the 'record' button, so I am removing
# these data which has 0 steering angle at a turn.
good_lines = lines[14:]
train_samples, validation_samples = train_test_split(good_lines, test_size=0.2)

# Model
model = Sequential()
model.add(Cropping2D(cropping=((64,32), (0,0)), input_shape=(160,320,3))) # Crop to the road
model.add(Lambda(lambda x: (x / 255.0) - 0.5)) # Normalize
model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dropout(rate=0.5))
model.add(Dense(100))
model.add(Dropout(rate=0.5))
model.add(Dense(50))
model.add(Dropout(rate=0.5))
model.add(Dense(10))
model.add(Dropout(rate=0.5))
model.add(Dense(1))

# Training Setup
epoch = 50
batch_size=32
filename = 'model'
# Data
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
# Callbacks
checkpoint = ModelCheckpoint(filepath=(filename+'.h5'), monitor='val_loss', save_best_only=True)
stopper = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5)
# Training
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, \
            steps_per_epoch=np.ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=np.ceil(len(validation_samples)/batch_size), \
            epochs=epoch, callbacks = [checkpoint, stopper])
# Save history
with open(filename+'.p', 'wb') as file_pi:
    pickle.dump(history_object.history, file_pi)
    
    
    
# # Other stuff I've tried    
#model.fit(X_train, y_train, validation_split=0.5, shuffle=True, epochs=epoch, callbacks=[checkpoint, stopper]) # loss: 0.05, vac_los: 0.4?
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epoch, callbacks=[checkpoint, stopper]) # loss: 0.05+, vac_loss: 0.11
# # Without callbacks
# epoch = 5
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epoch) 
# model.save('model_overfit.h5') # loss: 0.037, vac_loss: 0.12
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epoch)
# model.save('model_overfit2.h5') # - loss: 0.0339 - val_loss: 0.1243
# model.fit(X_train, y_train, shuffle=True, epochs=epoch)
# model.save('model_superoverfit.h5') # loss: 0.04