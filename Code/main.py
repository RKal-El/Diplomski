import os
import pandas
from sklearn.model_selection import train_test_split
import tensorflow


def load_data():
    data = pandas.read_csv(os.path.join(os.getcwd(), 'data', 'driving_log.csv'),
                           names=['center', 'left', 'right', 'steering',
                                  'throttle', 'break', 'speed'])

    return data


def build_graph():
    height = 5
    width = 7
    channels = 3
    n_inputs = height * width

    conv_fmaps = [24, 36, 48, 64, 64]
    conv_ksize =[5, 5, 5, 3, 3]
    conv_stride = [2, 2, 2, 1, 1]
    conv_pad = "VALID"

    n_fc = [100, 50, 10]
    dropout_rate = 0.5      # == 1 - keep_prob
    n_outputs = 1

    g = tensorflow.Graph()

    with g.as_default():
        with tensorflow.name_scope('inputs'):
            X = tensorflow.placeholder(tensorflow.float32, shape=[None,
                                                                  n_inputs],
                                       name='X')
            X_reshaped = tensorflow.reshape(X, shape=[-1, height, width,
                                                      channels])
            y = tensorflow.placeholder(tensorflow.float32, shape=[None],
                                       name='y')

        with tensorflow.name_scope('conv'):
            conv1 = tensorflow.layers.conv2d(X_reshaped, filters=conv_fmaps[0],
                                             kernel_size=conv_ksize[0],
                                             strides=conv_stride[0],
                                             padding=conv_pad,
                                             activation=tensorflow.nn.relu,
                                             name='conv1')
            conv2 = tensorflow.layers.conv2d(conv1, filters=conv_fmaps[1],
                                             kernel_size=conv_ksize[1],
                                             strides=conv_stride[1],
                                             padding=conv_pad,
                                             activation=tensorflow.nn.relu,
                                             name='conv2')
            conv2_flat = tensorflow.reshape(conv2, shape=[-1, conv_fmaps * 5
                                                          * 5])

        with tensorflow.name_scope('fc'):
            fc1 = tensorflow.layers.dense(conv2_flat, n_fc[0],
                                          activation=tensorflow.nn.relu,
                                          name='fc1')
            fc2 = tensorflow.layers.dense(fc1, n_fc[1],
                                          activation=tensorflow.nn.relu,
                                          name='fc2')
            fc3 = tensorflow.layers.dense(fc2, n_fc[2],
                                          activation=tensorflow.nn.relu,
                                          name='fc3')

        with tensorflow.name_scope('output'):
            prediction = tensorflow.layers.dense(fc3, n_outputs, name='output')

        with tensorflow.name_scope('train'):
            loss = tensorflow.losses.mean_squared_error(labels=y,
                                                        predictions=prediction)
            optimizer = tensorflow.train.AdamOptimizer()
            training_op = optimizer.minimize(loss)

        with tensorflow.name_scope('eval'):
            # correct = tensorflow.nn.in_top_k(prediction, y, 1) # ???
            # accuracy = tensorflow.losses.mean_squared_error() # ???
            eval_loss = tensorflow.losses.mean_squared_error(labels=y,
                                                             predictions=
                                                             prediction)

        with tensorflow.name_scope('init_and_save'):
            init = tensorflow.global_variables_initializer()
            saver = tensorflow.train.Saver()

    return g


def train_ann(graph, X_train, y_train, X_valid, y_valid):
    n_epochs = 60
    batch_size = 64
    with tensorflow.Session(graph=graph) as sess:
        feed = {'Placeholder:0': X_train, 'Placeholder:1': y_train}
        sess.run('training_op', feed_dict=feed)


def main():
    all_data = load_data()

    x = all_data[['center', 'left', 'right']].values
    y = all_data['steering'].values

    X_train, y_train, X_valid, y_valid = train_test_split(x, y, test_size=0.1)

    '''Preprocessing
    '''

    graph = build_graph()

    # train_ann(graph, X_train, y_train, X_valid, y_valid)


if __name__ == '__main__':
    main()
