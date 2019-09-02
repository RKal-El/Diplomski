''' Script for plot histograms of all data

Marko Rasetina
August 4, 2019
'''

import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn
import os

DATA_DIR = '.\data_csv'
PLOT_DIR = '.\plot_graph'
DRIVING_DATA = ['right_lane_original', 'left_lane_original',
                'left_lane_converted', 'right_lane_n_', 'left_lane_n_',
                'full_data_original', 'full_data_n_']
SHARP_TURNS = '_sharp_turns'
COLUMNS = ['center', 'left', 'right', 'steering', 'throttle', 'reverse',
           'speed']
N = 15
BINS = 21


def load_driving_data(data_dir, file_name, columns):
    dataframe = pandas.read_csv(os.path.join(data_dir, file_name),
                                names=columns)
    return dataframe


def save_fig(name):
    global PLOT_DIR

    print('Saving figure:\t', name, '\n')
    name = name + '.png'
    plt.savefig(os.path.join(PLOT_DIR, name), format='png', dpi=300)


def plot_histogram(data, name, bins=21, n=0, weighted=False, sharp=False):
    hist, bin_edges = numpy.histogram(data, bins=bins, range=(data.min(),
                                                              data.max()))

    if weighted:
        title = name[:-4] + f' za n = {n}'
    elif sharp:
        title = name[:-16] + f'_sharp_turns za n = {n}'
    else:
        title = name

    seaborn.set(color_codes=True)
    fig = plt.figure(figsize=(8, 5))
    fig.suptitle(title, fontsize=20)
    plt.subplots_adjust(left=0.12, right=0.88, bottom=0.12, top=0.88)

    plt.subplot(111)
    plt.hist(bin_edges[:-1], bins=bin_edges, weights=hist)

    plt.xlabel('Kut upada')
    plt.ylabel('Broj ponavljanja kuteva upada')

    save_fig(name)
    # plt.show()


def main():
    global DATA_DIR
    global DRIVING_DATA
    global N
    global BINS
    global SHARP_TURNS

    for data_type in DRIVING_DATA:
        if data_type[-1] == "_":
            for n in range(N):
                full_name = data_type + str(n + 1)
                file_name = full_name + '.csv'
                data = load_driving_data(DATA_DIR, file_name, COLUMNS)
                plot_data = data['steering']
                plot_histogram(plot_data, full_name, BINS, n + 1, True)
        else:
            full_name = data_type
            file_name = full_name + '.csv'
            data = load_driving_data(DATA_DIR, file_name, COLUMNS)
            plot_data = data['steering']
            plot_histogram(plot_data, full_name, BINS)

    for n in range(N):
        full_name = DRIVING_DATA[-1] + str(n + 1) + SHARP_TURNS
        file_name = full_name + '.csv'
        data = load_driving_data(DATA_DIR, file_name, COLUMNS)
        plot_data = data['steering']
        plot_histogram(plot_data, full_name, BINS, n + 1, sharp=True)


if __name__ == '__main__':
    main()
