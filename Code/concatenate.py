''' Script for concatenate driving log data (for left and right lane) before
steering angle is preprocessed and concatenate driving log data (for left and
right lane) after steering angle is processed with weighted mean.

Marko Rasetina
August 1, 2019
'''

import pandas
import os


DATA_DIRECTORY = 'P:\Fax\Diplomski\\4. semestar (' \
                 'Diplomski)\Diplomski\Code\data_csv'
CSV_DATA_FIRST_HALF = ['right_lane_original', 'right_lane_n_']
CSV_DATA_SECOND_HALF = ['left_lane_converted', 'left_lane_n_']
N = 15
CSV_FULL_DATA = ['full_data_original', 'full_data_n_']
SHARP_TURNS = 'sharp_turns'
COLUMNS = ['center', 'left', 'right', 'steering', 'throttle', 'reverse',
           'speed']


def read_csv_file(directory, name, columns):
    dataframe = pandas.read_csv(os.path.join(directory, name), names=columns)
    return dataframe


def concatenate_dataframes(first_half, second_half):
    df = pandas.concat([first_half, second_half], sort=False)
    return df


def save_into_csv(directory, data, name):
    data.to_csv(os.path.join(directory, name) + '.csv', encoding='utf-8',
                index=False, header=False)


def main():
    global DATA_DIRECTORY
    global CSV_DATA_FIRST_HALF
    global CSV_DATA_SECOND_HALF
    global N
    global CSV_FULL_DATA
    global COLUMNS
    global SHARP_TURNS

    for first_half, second_half, full_data in zip(CSV_DATA_FIRST_HALF,
                                                  CSV_DATA_SECOND_HALF,
                                                  CSV_FULL_DATA):
        if first_half == 'right_lane_original':
            df_f = read_csv_file(DATA_DIRECTORY, first_half + '.csv', COLUMNS)
            df_s = read_csv_file(DATA_DIRECTORY, second_half + '.csv', COLUMNS)
            concatenate_df = concatenate_dataframes(df_f, df_s)
            save_into_csv(DATA_DIRECTORY, concatenate_df, full_data)
        else:
            for n in range(N):
                df_f = read_csv_file(DATA_DIRECTORY, first_half + str(n + 1) +
                                     '.csv', COLUMNS)
                df_s = read_csv_file(DATA_DIRECTORY, second_half + str(n + 1)
                                     + '.csv', COLUMNS)
                concatenate_df = concatenate_dataframes(df_f, df_s)
                save_into_csv(DATA_DIRECTORY, concatenate_df, full_data +
                              str(n + 1))

    sharp_turns = read_csv_file(DATA_DIRECTORY, SHARP_TURNS + '.csv', COLUMNS)
    for n in range(N):
        name = CSV_FULL_DATA[1] + str(n + 1)
        data_without_sharp_turn = read_csv_file(DATA_DIRECTORY, name + '.csv',
                                                COLUMNS)
        concatenate_df = concatenate_dataframes(sharp_turns,
                                                data_without_sharp_turn)
        name = name + '_sharp_turns'
        save_into_csv(DATA_DIRECTORY, concatenate_df, name)


if __name__ == '__main__':
    main()
