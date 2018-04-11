import cv2
import csv
import os
import sklearn
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split

def preprocessing():
	# normalization + cropping
	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))
	model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
	return model

def loadLogs(path):
    # read each line from the csv log file and put them in a list
    lines = []
    with open(path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def nvidiaModel():
	# this is the nvidia model with 2 dropout layers
    model = preprocessing()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

def collectData(path):
    # collect the center, left, right images, and the corresponding angles (only for the center images)
    folder_names = [x[0] for x in os.walk(path)]
    img_paths = list(filter(lambda x: os.path.isfile(x + '/driving_log.csv'), folder_names))
    centerimg_all, leftimg_all, rightimg_all, meas_all = [], [], [], []
    for directory in img_paths:   # relative path
        lines = loadLogs(directory)
        centerimg, leftimg, rightimg, meas = [], [], [], []
        for line in lines:  # each line in the csv file
            centerimg.append(line[0].strip())
            leftimg.append(line[1].strip())
            rightimg.append(line[2].strip())
            meas.append(float(line[3]))
        centerimg_all.extend(centerimg)
        leftimg_all.extend(leftimg)
        rightimg_all.extend(rightimg)
        meas_all.extend(meas)
    return centerimg_all, leftimg_all, rightimg_all, meas_all # (meas is 1/3..)

def fixMultiCamers(center, left, right, measurement, correction):
    # fix the angles by adding correction value and make each image has one angle (for left and right images as well)
    new_paths, measurements = [], []
    new_paths.extend(center)
    new_paths.extend(left)
    new_paths.extend(right)
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])
    return new_paths, measurements

def myGenerator(samples, batch_size):
    # generate training sets by flipping each image on the fly
    num_samples = len(samples)
    while True:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            imgs = []
            steerings = []
            for imgpath, measurement in batch_samples:
                orgimg = cv2.imread(imgpath)
                img = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
                imgs.append(img)
                steerings.append(measurement)
                # generate Flipping imgs on the fly
                imgs.append(cv2.flip(img,1))
                steerings.append(-1.0 * measurement)
                inputs = np.array(imgs)
                outputs = np.array(steerings)
            yield sklearn.utils.shuffle(inputs, outputs)



if __name__ == '__main__':
    # main script, where correction value is set to 0.12, and we use 5 epoches.
    imgs = []
    measurements = []
    correction = 0.12
    batch_size = 32
    center_paths, left_paths, right_paths, measurements = collectData('data')
    imgpaths, measurements = fixMultiCamers(center_paths, left_paths, right_paths, measurements, correction)
    samples = list(zip(imgpaths, measurements))
    train_samples, validation_samples = train_test_split(samples, test_size = 0.3)
    train_generator = myGenerator(train_samples, batch_size)
    validation_generator = myGenerator(validation_samples, batch_size)
    model = nvidiaModel()
    model.compile(loss='mse', optimizer='adam')

    fit_history = model.fit_generator(train_generator, samples_per_epoch= \
                        len(train_samples), validation_data=validation_generator, \
                        nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

    model.save('model.h5')

    print(fit_history.history['loss'])
    print(fit_history.history['val_loss'])

