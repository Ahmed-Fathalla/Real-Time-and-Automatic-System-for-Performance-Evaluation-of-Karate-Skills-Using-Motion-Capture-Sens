from time import time
import csv
import pandas as pd
import numpy as np
from scipy import signal
import os
import matplotlib.pyplot as plt
from .time_utils import get_TimeStamp_str
from .skill import write_, get_files_cols_all_joints

time_stamp = ''
def create_Normalized_Trial(skill, folder_to_test, trial_path = "trial/"):
    global time_stamp
    trial_values = []
    files, cols, all_joints = get_files_cols_all_joints()

    index1 = -1
    draw = False
    for file in files:
        index1 = index1 + 1
        for col in cols[index1]:
            trial_v1 = []

            a = time()
            positions, df = readAngle_validation(trial_path, folder_to_test, file, col, 9, draw, skill)
            write_('readAngle ,'  + str(time() - a))

            number_of_files = len(df)
            temp_list = []
            for i in range(0, number_of_files):
                temp_list.append(df[i][0])

            trial_v1.append(min(temp_list))

            time_positions = []
            with open('skill_%d model/time-positions.csv'%skill, newline='') as f:
                reader = csv.reader(f)
                time_positions = list(reader)
            time_positions = time_positions[0]
            time_positions = [float(x) for x in time_positions]

            temp_list = []
            for j in time_positions:
                temp_list = []

                for i in range(0, number_of_files):
                    index = int(j * len(df[i]))
                    temp_list.append(df[i][index])

                trial_v1.append(min(temp_list))

            temp_list = []
            for i in range(0, number_of_files):
                temp_list.append(df[i][len(df[i]) - 1])
            trial_v1.append(min(temp_list))

            trial_values.append(trial_v1)

    for i in range(0, len(trial_values)):
        trial_values[i].pop(0)
        trial_values[i].pop()
        trial_values[i].insert(0, all_joints[i])

    temp = time_positions
    temp.insert(0, " ")
    trial_values.insert(0, temp)

    with open( trial_path + "trial skill_%d dir_%s %s.csv"%(skill,folder_to_test, time_stamp), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(trial_values)

def readAngle_validation(APP_FOLDER, folder_to_test, file, column, window_size, draw, skill):
    totalDir = 0
    reads = []
    dirs1 = []
    df_temp = pd.DataFrame()
    df = pd.DataFrame()

    col_counter = 0

    folders = []

    for base, dirs, files in os.walk('%s/%s'%(APP_FOLDER,folder_to_test)):
        dirs1.append(str(base))
        folders.append(str(base).split('/')[-1])

        path = str(base) + "/" + file
        new_path = str(base) + "/new-" + file

        with open(path, 'r') as f:
            with open(new_path, 'w') as f1:
                next(f)  # skip header line
                for line in f:
                    f1.write(line)

        df_temp = pd.read_csv(new_path)
        reads.append(df_temp[str(column)].tolist())
        df[col_counter] = df_temp[str(column)]
        col_counter = col_counter + 1

    positions = []
    for x1 in range(0, col_counter):
        arr = list()
        inv_arr = list()
        for v in reads[x1]:
            arr.append(v)
            inv_arr.append(-1 * v)

        data_x = np.arange(start=1, stop=len(arr) + 1, step=1, dtype='int')

        widths = np.arange(1, window_size)  # Widths range should cover the expected width of peaks of interest.

        a = time()
        peak_indexes = signal.find_peaks_cwt(arr, widths)
        write_( 'CWT_peaks -> %d,'%x1+ str(time()-a ) )

        max_value = max(arr)
        max_index = arr.index(max_value)
        if not (max_index in peak_indexes):
            np.append(peak_indexes, max_index)

        for i in range(0, len(peak_indexes)):
            start = max(0, peak_indexes[i] - 5)
            end = min(len(arr) - 1, peak_indexes[i] + 5)
            for k in range(start, end + 1):
                if (arr[k] > arr[peak_indexes[i]]):
                    peak_indexes[i] = k
        positions.append(peak_indexes)
        peak_x = peak_indexes

        peak_y = []
        for i in range(0, len(peak_indexes)):
            peak_y.append(arr[peak_indexes[i]])
        a = time()
        valley_indexes = signal.find_peaks_cwt(inv_arr, widths)
        write_('CWT_valley -> %d,'%x1 + str(time() - a))

        max_value = max(inv_arr)
        max_index = inv_arr.index(max_value)
        if not (max_index in valley_indexes):
            np.append(valley_indexes, max_index)
        for i in range(0, len(valley_indexes)):
            start = max(0, valley_indexes[i] - 5)
            end = min(len(inv_arr) - 1, valley_indexes[i] + 5)
            for k in range(start, end):
                if (inv_arr[k] > inv_arr[valley_indexes[i]]):
                    valley_indexes[i] = k
        positions.append(valley_indexes)

        # Plot valleys
        valley_x = valley_indexes
        valley_y = []
        for i in range(0, len(valley_indexes)):
            valley_y.append(arr[valley_indexes[i]])

        if draw == True:
            (fig, ax) = plt.subplots()
            ax.plot(data_x, arr)
            ax.plot(peak_x, peak_y, marker='o', linestyle='dashed', color='green', label="Peaks")
            ax.plot(valley_x, valley_y, marker='o', linestyle='dashed', color='red', label="Valleys")

            # Save graph to file.
            string1 = file + "-" + column + " trial-" + folders[x1]
            plt.title(string1)
            plt.legend(loc='best')
            file_name = "peaks" + str(x1) + "-trial-" + folders[x1] + ".png"
            # print(file_name)
            col_updated = column.replace("/", "-")

            outdir = ''
            path_out = outdir + file + "/" + col_updated + "/"
            if not (os.path.exists(path_out)):
                os.makedirs(path_out)
            plt.savefig(path_out + file_name)
            plt.clf()
            # plt.show()

    return positions, reads


def readModel(path):
    df = pd.read_csv(path)
    df.drop(df.iloc[:, :1], inplace=True, axis=1)
    result = df.values.tolist()
    return result


def Calling_1(skill, trial_path, folder_to_test, normalize_only=False):
    a = time()
    create_Normalized_Trial(skill, folder_to_test, trial_path=trial_path)  # eeee trial_path value
    write_('creat_Normalized_Trial,' + str(time() - a), record_time=(not normalize_only))

    if normalize_only:
        return ['']*4

    model_paths = [ "skill_%d model/min.csv"%skill, "skill_%d model/max.csv"%skill,  "skill_%d model/std.csv"%skill, "skill_%d model/mean.csv"%skill  ]
    model_min = readModel(model_paths[0])
    model_max = readModel(model_paths[1])
    model_std = readModel(model_paths[2])
    model_avg = readModel(model_paths[3])

    return model_min, model_max, model_std, model_avg

def run_validation(skill, folder_to_test, trial_path = "trial/", time_stamp_flag=True, normalize_only=False):
    a = time()
    global time_stamp
    time_stamp = '' if not time_stamp_flag else ' '+get_TimeStamp_str()

    trial_path = trial_path+'skill_%d/'%skill
    model_min, model_max, model_std, model_avg = Calling_1(skill, trial_path, folder_to_test, normalize_only)

    if normalize_only:
        return ''

    files, cols, all_joints = get_files_cols_all_joints()
    trial_data = readModel(trial_path + "trial skill_%d dir_%s%s.csv"%(skill, folder_to_test, time_stamp))  # eeee

    results = []
    for rowID in range(0, len(trial_data)):
        result = []
        for colID in range(0, len(trial_data[0])):

            if trial_data[rowID][colID] == model_avg[rowID][colID]:
                result.append(1)
            elif (trial_data[rowID][colID] <= (model_avg[rowID][colID] + model_std[rowID][colID])) and (
                    trial_data[rowID][colID] >= (model_avg[rowID][colID] - model_std[rowID][colID])):
                result.append(2)
            elif (trial_data[rowID][colID] <= (model_max[rowID][colID])) and (
                    trial_data[rowID][colID] >= (model_min[rowID][colID])):
                result.append(3)
            else:
                result.append(4)
        results.append(result)

    for rowID in range(0, len(results)):
        scores = [0, 0, 0, 0]

        for colID in range(0, len(results[rowID])):
            index = int(results[rowID][colID]) - 1
            scores[index] += 1

        max1 = 0
        maxID = 0
        for i in range(len(scores) - 1, -1, -1):
            if scores[i] > max1:
                maxID = i
                max1 = scores[i]
        results[rowID].append("<>")
        results[rowID].append(maxID + 1)
    ##########################################################

    row_results = []
    for colID in range(0, len(results[0]) - 2):
        scores = [0, 0, 0, 0]

        for rowID in range(0, 39):
            index = int(results[rowID][colID]) - 1
            scores[index] += 1

        max1 = 0
        maxID = 0
        for i in range(len(scores) - 1, -1, -1):
            if scores[i] > max1:
                maxID = i + 1
                max1 = scores[i]
        row_results.append(maxID)

    scores = [0, 0, 0, 0]
    for j in range(0, len(row_results)):
        scores[row_results[j]] += 1
    max1 = 0
    maxID = 0
    for i in range(len(scores) - 1, -1, -1):
        if scores[i] > max1:
            maxID = i
            max1 = scores[i]

    final = ["Final Result: "]
    final.append(maxID)

    results.append([""])
    row_results.insert(0, "")
    row_results.insert(0, "")
    results.append(row_results)

    for rowID in range(0, len(all_joints)):
        string1 = all_joints[rowID].split(" ")

        results[rowID].insert(0, string1[1])
        results[rowID].insert(0, string1[0])

    time_positions = []
    with open('skill_%d model/time-positions.csv'%skill, newline='') as f:
        reader = csv.reader(f)
        time_positions = list(reader)
    time_positions = time_positions[0]
    time_positions = [float(x) for x in time_positions]

    time_positions.insert(0, "")
    time_positions.insert(0, "")
    results.insert(0, time_positions)
    results.append("")

    results.append(final)

    with open(trial_path+"results skill_%d dir_%s%s.csv"%(skill, folder_to_test, time_stamp), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)

    write_('run_validation, ' + str(time() - a))

    with open('time.txt', 'r') as myfile:
        a = myfile.read(  )
    a = a.split('\n')[:-1]
    df = pd.DataFrame(np.array(a).reshape(-1,1), columns=['a'])
    df['method'], df['time'] = zip(*df['a'].apply(lambda x: x.split(',')))
    df[['method','time']].to_csv(trial_path+"Profling_val skill_%d dir_%s%s.csv"%(skill, folder_to_test, time_stamp), index=False)
    os.remove("time.txt")
