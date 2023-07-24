import csv
import pandas as pd
import io
import matplotlib

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os

def readAngle(file, column, window_size, check_w = 5):  
    APP_FOLDER = 'data/'
    totalDir = 0

    reads = []

    dirs1 = []
    df_temp = pd.DataFrame()
    df = pd.DataFrame()

    col_counter = 0
    for base, dirs, files in os.walk(APP_FOLDER):
        if len(files)==0:continue   

        dirs1.append(str(base))
        df_temp = pd.read_csv(  str(base) + "/" + file,
                                skiprows=1) 

        reads.append(df_temp[ str(column) ].tolist())
        df[col_counter] = df_temp[ str(column) ]
        col_counter += 1

    positions = []
    # for x1 in range(0, col_counter):
    for x1, rec in enumerate(reads):
        arr = rec #        fathalla_1
        inv_arr = (np.array(arr)*-1).tolist() 

        data_x = np.arange(start=1, stop=len(arr) + 1, step=1, dtype='int')
        widths = np.arange(1, window_size) 


        # 1
        peak_indexes = signal.find_peaks_cwt(arr, widths)

        # 2
        # peaks = scipy.signal.argrelextrema( np.array(arr), comparator=np.greater, order=5)
        # peak_indexes = peaks[0]

        # 3
        # peak_indexes,_ = scipy.signal.find_peaks(arr, height=7, distance=2.1)

        # 4
        # peak_indexes = peakutils.peak.indexes(np.array(arr), min_dist=4)

        # 5
        # registration function

        max_value = max(arr)
        max_index = arr.index(max_value)
        if not (max_index in peak_indexes):
            np.append(peak_indexes, max_index)

        for i in range(0, len(peak_indexes)):
            start = max(0, peak_indexes[i] - check_w)
            end = min(len(arr) - 1, peak_indexes[i] + check_w)
            for k in range(start, end + 1):
                if arr[k] > arr[peak_indexes[i]]:
                    peak_indexes[i] = k

        (fig, ax) = plt.subplots()
        ax.plot(data_x, arr)
        peak_x = peak_indexes
        peak_y = np.array(arr)[peak_indexes] 

        ax.plot(peak_x, peak_y, marker='o', linestyle='dashed', color='green', label="Peaks")

        valley_indexes = signal.find_peaks_cwt(inv_arr, widths)

        max_value = max(inv_arr)
        max_index = inv_arr.index(max_value)
        if not (max_index in valley_indexes):
            np.append(valley_indexes, max_index)

        for i in range(0, len(valley_indexes)):
            start = max(0, valley_indexes[i] - check_w)
            end = min(len(inv_arr) - 1, valley_indexes[i] + check_w)
            for k in range(start, end):
                if inv_arr[k] > inv_arr[valley_indexes[i]]:
                    valley_indexes[i] = k

        positions.append(peak_indexes)
        positions.append(valley_indexes)

        # Plot valleys
        valley_x = valley_indexes
        valley_y = np.array(arr)[valley_indexes]
        ax.plot(valley_x, valley_y, marker='o', linestyle='dashed', color='red', label="Valleys")

        # Save graph to file.
        plt.title('Find peaks and valleys using find_peaks_cwt()')
        plt.legend(loc='best')
        file_name = "peaks" + str(x1) + ".png"
        print(file_name)
        plt.savefig(file_name)
        plt.show()

    return positions, reads