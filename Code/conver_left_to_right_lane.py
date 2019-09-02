''' Script for converting left side driving log data to look like it has been
driven from right side

Marko Rasetina
August 2, 2019
'''

import pandas
import os
import cv2


SAVE_TO_IMAGES = 'P:\DataSelfDrivingCar\left_side\converted_to_right_side'
LOG_DATA_PATH = '.\data_csv'
LOG_DATA = 'left_lane_original.csv'
PROCESSED_DATA = 'left_lane_converted.csv'
COLUMNS = ['center', 'left', 'right', 'steering', 'throttle', 'reverse',
           'speed']


def read_data(log_data_path, log_data, columns):
    dataframe = pandas.read_csv(os.path.join(log_data_path, log_data),
                                names=columns)
    return dataframe


def extract_img_name(path):
    return os.path.basename(path)


def flip_image(images_path, path_to_save):
    for path in images_path:
        image = cv2.imread(path)
        left_flip = cv2.flip(image, flipCode=180)
        image_name = extract_img_name(path)
        cv2.imwrite(os.path.join(path_to_save, image_name), left_flip)


def flip_and_save(csv_file, path, columns):
    new_df_right_lane = pandas.DataFrame()
    path_dict = {}
    path_dict[columns[0]] = csv_file.iloc[:, 0]
    path_dict[columns[1]] = csv_file.iloc[:, 1]
    path_dict[columns[2]] = csv_file.iloc[:, 2]

    for key in path_dict.keys():
        flip_image(path_dict[key], path)
        if key == 'center':
            new_df_right_lane[key] = path_dict[key]
        elif key == 'left':
            new_df_right_lane['right'] = path_dict[key]
        elif key == 'right':
            new_df_right_lane['left'] = path_dict[key]

    return new_df_right_lane


def correct_steering_angle(steering_angle_left):
    steering_angle_right = pandas.DataFrame()

    steering_angle_right['steering'] = -1 * steering_angle_left

    return steering_angle_right


def csv_to_save(right_lane_images, rest_of_data, path, name):
    df_for_saving = pandas.DataFrame()

    df_for_saving['center'] = right_lane_images.iloc[:, 0]
    df_for_saving['left'] = right_lane_images.iloc[:, 2]
    df_for_saving['right'] = right_lane_images.iloc[:, 1]

    steering_angle = correct_steering_angle(rest_of_data.iloc[:, 0])
    df_for_saving['steering'] = steering_angle.iloc[:, 0]

    df_for_saving['throttle'] = rest_of_data.iloc[:, 1]
    df_for_saving['reverse'] = rest_of_data.iloc[:, 2]
    df_for_saving['speed'] = rest_of_data.iloc[:, 3]

    df_for_saving.to_csv(os.path.join(path, name), encoding='utf-8',
                         header=False, index=False)


def main():
    global SAVE_TO_IMAGES
    global LOG_DATA_PATH
    global LOG_DATA
    global COLUMNS
    global PROCESSED_DATA

    read_csv_file = read_data(LOG_DATA_PATH, LOG_DATA, COLUMNS)

    right_lane = flip_and_save(read_csv_file, SAVE_TO_IMAGES, COLUMNS[:3])

    csv_to_save(right_lane, read_csv_file.iloc[:, 3:], LOG_DATA_PATH,
                PROCESSED_DATA)


if __name__ == '__main__':
    main()
