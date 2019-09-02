''' Script for driving car in autonomous mode

Marko Rasetina
August 1, 2019
'''

import socketio
from flask import Flask
from PID_Controller import PIDController
import argparse
import h5py
from keras import __version__ as keras_version
from keras.models import load_model
import os
import shutil
import eventlet.wsgi
import eventlet
from PIL import Image
from io import BytesIO
import base64
import numpy
import cv2
import sys
from datetime import datetime
import warnings
import tensorflow


sio = socketio.Server()
app = Flask(__name__)
controller = PIDController(0.7, 0.001, 0.4)     # Kp, Ki, Kd
speed = 25.2    # 15.1, 20.2, 25.2
controller.set_speed(speed)


@sio.on('connect')
def connect(sid, environment):
    print('Connect ',  sid)
    send_control(0, 0)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data['steering_angle']
        # The current throttle of the car
        throttle = data['throttle']
        # The current speed of the  car
        speed = data['speed']
        # The current  image from the center camera of the car
        image_string = data['image']

        image = Image.open(BytesIO(base64.b64decode(image_string)))
        image_array = numpy.asarray(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        steering_angle = float(model.predict(image_array[None, :, :, :],
                                             batch_size=1))

        throttle = controller.update(float(speed))

        print(f'{steering_angle:.3f}\t{throttle:.2f}', flush=True)

        sys.stdout.flush()
        send_control(steering_angle, throttle)

        # Save  frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save(f'{image_filename}.jpg')

    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


def send_control(steering_angle, throttle):
    sio.emit('steer',
             data={
                 'steering_angle': steering_angle.__str__(),
                 'throttle': throttle.__str__()
             },
             skip_sid=True)


def warn(*args, **kwargs):
    pass


if __name__ == '__main__':
    # Remove warnings and tensorflow system messages
    warnings.warn = warn
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tensorflow.compat.v1.logging.set_verbosity(
        tensorflow.compat.v1.logging.FATAL)

    model = None

    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model',
                        type=str,
                        help='Path to model h5 file. Model should be on the '
                             'same path.')
    parser.add_argument('image_folder',
                        type=str,
                        nargs='?',
                        default='',
                        help='Path to image folder. This is where the images '
                             'from the run will be saved.')
    args = parser.parse_args()

    # Check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf-8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version, ', but the model '
              'was built using ', model_version)

    model = load_model(args.model)

    if args.image_folder != '':
        print('Creating image folder at {}'.format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print('RECORDING THIS RUN...')
    else:
        print('NOT RECORDING THIS RUN...')

    # Wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # Deploy  as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
