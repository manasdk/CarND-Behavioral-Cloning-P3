import csv
import numpy as np
import tensorflow as tf
from keras.layers import Activation, Cropping2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from scipy import ndimage
from scipy.misc import imresize
from sklearn.model_selection import train_test_split


TEST_DATA_DIR = './test_data'
DRIVING_LOG_FILE = 'driving_log.csv'

# tunable params
STEERING_CORRECTION = 0.1
LEARNING_RATE = 0.0001


def create_model():
    """
    Based on the recommendations https://carnd-forums.udacity.com/questions/26214464/behavioral-cloning-cheatsheet
    recreating the model from https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    model = Sequential()
    # normalization as per recommendation in project help
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(32,64,3)))
    model.add(Cropping2D(cropping=((10,4), (0,0)), input_shape=(32,64,3)))

    # 5 convolution layers
    model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 5, 5, border_mode='same', subsample=(1, 1), activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1), activation='relu'))

    model.add(Flatten())

    # 4 fully connected and 1 output
    model.add(Dense(1164))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.summary()

    return model

def process_image(img_path):
    """
    resize the image to keep it manageable, else the load times were starting to get out of hand
    """
    image = ndimage.imread(img_path).astype(np.float32)
    # reduce size by 5x
    image = imresize(image, (32,64,3))
    return image


def read_datum(datum_row, test_data_dir=TEST_DATA_DIR, steering_correction=STEERING_CORRECTION):
    """
    Load all 3 images for each angle and also compensate for camera angle
    """
    center_img_path = test_data_dir + '/' + datum_row[0]
    center_img = process_image(center_img_path)
    center_steering_angle = float(datum_row[3])

    left_img_path = test_data_dir + '/' + datum_row[1]
    left_img = process_image(left_img_path)
    left_steering_angle = center_steering_angle + STEERING_CORRECTION

    right_img_path = test_data_dir + '/' + datum_row[2]
    right_img = process_image(right_img_path)
    right_steering_angle = center_steering_angle - STEERING_CORRECTION

    return (center_img, center_steering_angle), (left_img, left_steering_angle), (right_img, right_steering_angle)


def read_train_data(test_data_dir=TEST_DATA_DIR, driving_log_file=DRIVING_LOG_FILE):
    driving_log_file_path = test_data_dir + '/' + driving_log_file
    with open(driving_log_file_path, 'r') as driving_log_file_csv:
        X_train = []
        y_train = []
        reader = csv.reader(driving_log_file_csv, delimiter=',')
        for row in reader:
            c, l, r = read_datum(datum_row=row)
            X_train.append(c[0])
            y_train.append(c[1])
            X_train.append(l[0])
            y_train.append(l[1])
            X_train.append(r[0])
            y_train.append(r[1])
        print('### done reading rows returning values')
        return np.array(X_train), np.array(y_train)

def main():
    print('## read train data')
    X_train, y_train = read_train_data()

    print('## split train data')
    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.2)

    print('## create model and compile')
    model = create_model()
    adam_optimizer = Adam(LEARNING_RATE)
    model.compile(optimizer=adam_optimizer, loss='mse')

    print('## fit model, time for a coffee :)')
    model.fit(
        X_train,
        y_train,
        batch_size=100,
        validation_data=(X_val, y_val)
    )
    print('## model training done')

    print('## saving trained model to model.h5')
    model.save('model.h5')

main()
