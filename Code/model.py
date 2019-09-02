''' Script for training CNN model

Marko Rasetina
August 8, 2019
'''

import tensorflow
from datetime import datetime
import os
import pandas
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Conv2D, Flatten, Dense, Dropout
from keras.layers import MaxPool2D
from keras.layers import LeakyReLU
from tensorflow.python.keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import numpy
from augment import augment_data, load_image
import warnings


flags = tensorflow.app.flags
FLAGS = flags.FLAGS

# Command line flags
flags.DEFINE_string('output_dir', '.\\training_output\\', 'Output path for '
                                                          'trained model and '
                                                          'training '
                                                          'information')
flags.DEFINE_string('driving_log_dir', '.\\data\\', 'Path to simulator '
                                                    'driving log directories')
flags.DEFINE_integer('weights', 10, 'Number of frames taken to calc weighted '
                                    'mean to adjust steering angle')
flags.DEFINE_integer('epochs', 36, 'Number of epochs to train')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('train_size', 49208, 'Size of training data')
flags.DEFINE_integer('steps_per_epoch', (FLAGS.train_size //
                                         FLAGS.batch_size), 'Number of steps')
flags.DEFINE_float('test_size', 0.15, 'Size of test size')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate of the model')
flags.DEFINE_integer('height', 160, 'Height of image')
flags.DEFINE_integer('width', 320, 'Width of image')
flags.DEFINE_integer('channels', 3, 'Number of channels of image')


# def create_output_dir():
#     training_name = datetime.now().strftime('%Y%m%d-%H%M%S')
#     training_dir = os.path.join(FLAGS.output_dir, training_name)
#     assert (not os.path.exists(training_dir))
#     os.makedirs(training_dir)
#     print(f'Training output goes to {training_dir}', flush=True)
#
#     return training_dir


def load_data():
    data_df = pandas.read_csv(os.path.join(FLAGS.driving_log_dir,
                                           'driving_log.csv'), names=[
        'center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          test_size=
                                                          FLAGS.test_size)

    return X_train, X_valid, y_train, y_valid


def build_model(name, input_shape=(160, 320, 3)):
    model = Sequential()

    # Preprocess incoming data, centered around zero with small standard
    # deviation
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))

    # Remove some parts of the sky/background and the vehicle hood
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(48, (3, 3), strides=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(48, (3, 3), strides=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(MaxPool2D(pool_size=(1, 2)))

    model.add(Flatten())

    model.add(Dense(196))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.2))

    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.2))

    model.add(Dense(16))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.2))

    model.add(Dense(1, activation='tanh'))

    model.summary()

    tensorboard = TensorBoard(log_dir=f'.\logs\{name}', write_images=True,
                              write_graph=False)

    return model, tensorboard


def batch_generator(data_dir, img_paths, steering_angles, batch_size,
                    is_training):
    images = numpy.empty([batch_size, FLAGS.height, FLAGS.width,
                          FLAGS.channels])
    steers = numpy.empty(batch_size)

    while True:
        i = 0
        for index in numpy.random.permutation(img_paths.shape[0]):
            center, left, right = img_paths[index]
            steering_angle = steering_angles[index]

            # Augment data
            if is_training and numpy.random.rand() < 0.6:
                image, steering_angle = augment_data(data_dir, center, left,
                                                     right, steering_angle)
            else:
                image = load_image(data_dir, center)

            # Add the images and steering angle to the batch
            images[i] = image
            steers[i] = steering_angle

            i += 1
            if i == batch_size:
                break

        yield images, steers


def train_model(model, tensorboard, name, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint(f'{name}-'+'{epoch:03d}.h5',
                                 monitor='val_loss')

    model.compile(loss='msle',
                  optimizer=Adam(lr=FLAGS.learning_rate),
                  metrics=['mae', 'mse', 'mape', 'cosine'])

    model.fit_generator(batch_generator(FLAGS.driving_log_dir, X_train,
                                        y_train, FLAGS.batch_size, True),
                        steps_per_epoch=FLAGS.steps_per_epoch,
                        epochs=FLAGS.epochs,
                        max_queue_size=1,
                        validation_data=batch_generator(
                            FLAGS.driving_log_dir, X_valid, y_valid,
                            FLAGS.batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint, tensorboard],
                        verbose=2)


def warn(*args, **kwargs):
    pass


def main(_):

    # Remove warnings and tensorflow system messages
    warnings.warn = warn
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tensorflow.compat.v1.logging.set_verbosity(
        tensorflow.compat.v1.logging.FATAL)

    # output_path = create_output_dir()

    # Load data
    data = load_data()

    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    name = f'm_3_msle_n_{FLAGS.weights}_tanh_12_{now}'

    model, tensorboard = build_model(name=name)

    # serialize model to JSON
    model_json = model.to_json()
    with open(name + '.json', 'w') as json_file:
        json_file.write(model_json)

    train_model(model, tensorboard, name, *data)


if __name__ == '__main__':
    tensorflow.compat.v1.app.run()
