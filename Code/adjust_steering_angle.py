''' Script for adjusting steering angle using weighted mean

Marko Rasetina
August 2, 2019
'''

import numpy
import pandas
import os


DATA_DIR = '.\data_csv'
FILES_TO_CONVERT = ['left_lane_converted.csv', 'right_lane_original.csv']
LANES = ['left_lane', 'right_lane']
COLUMN_NAME = ['center', 'left', 'right', 'steering', 'throttle', 'reverse',
               'speed']
N_POINTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
WEIGHTS = numpy.asarray([1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169,
                         196, 225, 256])


def load_driving_data(data_dir, file, columns):
    dataframe = pandas.read_csv(os.path.join(data_dir, file), names=columns)
    return dataframe


def calc_weighted_mean_steering_angle(data, weights, n):
    result = 0
    weights_length = len(weights)
    norm_weights = numpy.zeros(weights_length)
    sum_of_weights = numpy.sum(weights)

    for i in range(weights_length):
        norm_weights[i] = weights[i] / sum_of_weights

    norm_weights = numpy.flip(norm_weights)

    for i in range(n + 1):
        result += (data[i] * norm_weights[i])

    return result


def process_driving_log_data(dataframe, n, weights, steering=''):
    post_processed_data = dataframe.iloc[:-n, :3].copy()
    np_array_of_data = dataframe[steering].values

    new_data = numpy.zeros((np_array_of_data.shape[0] - n), dtype=float)
    iteration = len(new_data)

    for i in range(iteration):
        data = np_array_of_data[i:n + 1 + i]
        result = calc_weighted_mean_steering_angle(data, weights, n)
        new_data[i] = result

    post_processed_data[steering] = new_data
    post_processed_data.reset_index(drop=True, inplace=True)

    return post_processed_data


def save_df_to_csv(data, key, data_dir):
    file_name = str(key[0]) + '_n_' + str(key[1]) + '.csv'
    csv_name = os.path.join(data_dir, file_name)
    data.to_csv(csv_name, index=False, header=False)


def main():
    global DATA_DIR
    global FILES_TO_CONVERT
    global LANES
    global COLUMN_NAME
    global N_POINTS
    global WEIGHTS

    driving_log_collection = {}
    processed_driving_log = {}

    for file, key in zip(FILES_TO_CONVERT, LANES):
        # key, _ = file.split('.')
        driving_log_collection[key] = load_driving_data(DATA_DIR, file,
                                                        COLUMN_NAME)

    for key in driving_log_collection.keys():
        for n in N_POINTS:
            df_to_process = driving_log_collection[key]
            processed_driving_log[(key, n)] = process_driving_log_data(
                df_to_process, n, WEIGHTS[:n+1], 'steering')

    for key in processed_driving_log.keys():
        save_df_to_csv(processed_driving_log[key], key, DATA_DIR)


if __name__ == '__main__':
    main()
