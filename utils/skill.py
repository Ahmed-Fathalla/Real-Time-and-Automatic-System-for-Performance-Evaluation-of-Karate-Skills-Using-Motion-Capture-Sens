import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import os

import numpy as np
import csv

from time import time

skill_dict = {
                1:'GEDAN BARAI',
                2:'OI ZUKI',
                3:'SOTO UKE',
                4:'AGE UKE',
            }

def write_(s, record_time=True):
    if record_time:
        with open('time.txt', 'a') as myfile:
            myfile.write( s + '\n' )

def get_files_cols_all_joints():
    files = ['left_ankle_angle.csv',
             'left_elbow_angle.csv',
             'left_hip_angle.csv',
             'left_knee_angle.csv',
             'left_shoulder_angle.csv',
             'left_wrist_angle.csv',
             'lumbar_joint_angle.csv',
             'neck_joint_angle.csv',
             'right_ankle_angle.csv',
             'right_elbow_angle.csv',
             'right_hip_angle.csv',
             'right_knee_angle.csv',
             'right_shoulder_angle.csv',
             'right_wrist_angle.csv'
             ]
    cols = [['left_ankle Plantar/Dorsi', 'left_ankle Int/Ext', 'left_ankle Sup/Pro'],
            ['left_elbow Flex/Ext', 'left_elbow Int/Ext'],
            ['left_hip Flex/Ext', 'left_hip Abd/Add', 'left_hip Int/Ext'],
            ['left_knee Flex/Ext', 'left_knee Varus/Valgus', 'left_knee Int/Ext'],
            ['left_shoulder Flex/Ext', 'left_shoulder Abd/Add', 'left_shoulder Int/Ext'],
            ['left_wrist Abd/Add', 'left_wrist Flex/Ext', 'left_wrist Int/Ext'],
            ['lumbar_joint Bend Front/Back', 'lumbar_joint Twist Left/Right', 'lumbar_joint Lean Right/Left'],
            ['neck_joint Bend Front/Back', 'neck_joint Twist Left/Right'],
            ['right_ankle Plantar/Dorsi', 'right_ankle Int/Ext', 'right_ankle Sup/Pro'],
            ['right_elbow Flex/Ext', 'right_elbow Int/Ext'],
            ['right_hip Flex/Ext', 'right_hip Abd/Add', 'right_hip Int/Ext'],
            ['right_knee Flex/Ext', 'right_knee Varus/Valgus', 'right_knee Int/Ext'],
            ['right_shoulder Flex/Ext', 'right_shoulder Abd/Add', 'right_shoulder Int/Ext'],
            ['right_wrist Abd/Add', 'right_wrist Flex/Ext', 'right_wrist Int/Ext']
            ]

    counter = 1
    index1 = 0
    all_joints = []
    for file in files:
        for col in cols[index1]:
            # print(str(counter) + " " + file + " " + col)
            all_joints.append(file + " " + col)
            counter = counter + 1
        index1 = index1 + 1

    return files, cols, all_joints

def get_skill_data(skill=1):
    outdir = ""
    APP_FOLDER = 'Data Prepairation/'

    discard_joints = disrcard_trial = None
    # enter your choice here
    if skill == 1:
        # skill 1  gedan
        skill_1_joints = [['left_wrist_angle.csv', 'left_wrist Flex/Ext'],
                          ['left_ankle_angle.csv', 'left_ankle Int/Ext'],
                          ['left_hip_angle.csv', 'left_hip Flex/Ext'],
                          ['left_knee_angle.csv', 'left_knee Varus/Valgus'],
                          ['right_shoulder_angle.csv', 'right_shoulder Abd/Add'],
                          ['right_hip_angle.csv', 'right_hip Abd/Add'],
                          ['right_hip_angle.csv', 'right_hip Int/Ext'],
                          ]
        skill_1_disrcard_trials = [[1, 9, 10, 16, 18, 20, 4, 51, 53],
                                   [5, 6, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                                   [8, 10, 12, 15, 17],
                                   [27],
                                   [2, 12],
                                   [12, 26],
                                   [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
                                   ]
        discard_joints = skill_1_joints
        disrcard_trial = skill_1_disrcard_trials
        outdir = "output" + "-Gedan" + "/"
        APP_FOLDER += 'GEDAN BARAI/'

    if skill == 2:
        # skill 2  Oi Zuki
        skill_2_joints = [['left_ankle_angle.csv', 'left_ankle Int/Ext'],
                          ['left_wrist_angle.csv', 'left_wrist Flex/Ext'],
                          ['right_ankle_angle.csv', 'right_ankle Int/Ext'],
                          ['right_elbow_angle.csv', 'right_elbow Flex/Ext'],
                          ['right_shoulder_angle.csv', 'right_shoulder Flex/Ext'],
                          ['right_knee_angle.csv', 'right_knee Varus/Valgus'],
                          ['right_knee_angle.csv', 'right_knee Int/Ext'],
                          ['right_shoulder_angle.csv', 'right_shoulder Abd/Add'],
                          ['right_wrist_angle.csv', 'right_wrist Abd/Add'],
                          ['right_wrist_angle.csv', 'right_wrist Flex/Ext'],
                          ['right_wrist_angle.csv', 'right_wrist Int/Ext']
                          ]
        skill_2_disrcard_trials = [[3, 5, 8, 9, 10, 12],
                                   [19, 20, 21, 36, 37, 38, 39, 40, 46, 48],
                                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 18, 19, 21, 24, 25, 26, 27, 29, 30, 31,
                                    32, 33, 34, 35, 40, 41],
                                   [3, 5, 7, 12, 15, 16, 21, 32, 34, 35, 36, 38, 39, 40, 44, 45, 46, 48, 49],
                                   [3, 5, 7, 12, 15, 16, 21, 32, 34, 35, 36, 38, 39, 40, 44, 45, 46, 48, 49],
                                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 35],
                                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 35],
                                   [34, 39],
                                   [30],
                                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                    25,
                                    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 40],
                                   [30]
                                   ]
        discard_joints = skill_2_joints
        disrcard_trial = skill_2_disrcard_trials
        outdir = "output" + "-Oi-Zuki" + "/"
        APP_FOLDER += 'OI ZUKI/'

    if skill == 3:
        # skill 3:  Soto Uke
        skill_3_joints = [['left_ankle_angle.csv', 'left_ankle Int/Ext'],
                          ['left_shoulder_angle.csv', 'left_shoulder Abd/Add'],
                          ['left_shoulder_angle.csv', 'left_shoulder Flex/Ext'],
                          ['left_wrist_angle.csv', 'left_wrist Flex/Ext'],
                          ['right_ankle_angle.csv', 'right_ankle Int/Ext'],
                          ['right_hip_angle.csv', 'right_hip Flex/Ext'],
                          ['right_hip_angle.csv', 'right_hip Abd/Add'],
                          ['right_shoulder_angle.csv', 'right_shoulder Flex/Ext'],
                          ['right_wrist_angle.csv', 'right_wrist Flex/Ext']
                          ]
        skill_3_disrcard_trials = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 22, 23, 27],
                                   [6, 8, 27, 29],
                                   [6, 8, 27, 29],
                                   [19, 20, 22, 23, 35, 37],
                                   [1, 14, 15, 16, 17, 22, 27, 28, 29],
                                   [14, 27],
                                   [14, 27],
                                   [14, 18, 26, 27, 28, 29, 30, 31, 32, 33, 35],
                                   [1, 5, 6, 7, 8, 9, 10, 12, 14, 15, 22, 29, 36, 37]
                                   ]

        discard_joints = skill_3_joints
        disrcard_trial = skill_3_disrcard_trials
        outdir = "output" + "Soto-Uke" + "/"
        APP_FOLDER += 'SOTO UKE/'

    if skill == 4:
        # skill 4:  Age Uke
        skill_4_joints = [['left_ankle_angle.csv', 'left_ankle Int/Ext'],
                          ['left_elbow_angle.csv', 'left_elbow Flex/Ext'],
                          ['left_hip_angle.csv', 'left_hip Abd/Add'],
                          ['left_shoulder_angle.csv', 'left_shoulder Flex/Ext'],
                          ['left_shoulder_angle.csv', 'left_shoulder Abd/Add'],
                          ['left_shoulder_angle.csv', 'left_shoulder Int/Ext'],
                          ['left_wrist_angle.csv', 'left_wrist Abd/Add'],
                          ['left_wrist_angle.csv', 'left_wrist Flex/Ext'],
                          ['right_ankle_angle.csv', 'right_ankle Int/Ext'],
                          ['right_hip_angle.csv', 'right_hip Abd/Add'],
                          ['right_knee_angle.csv', 'right_knee Varus/Valgus'],
                          ['right_shoulder_angle.csv', 'right_shoulder Flex/Ext'],
                          ['right_wrist_angle.csv', 'right_wrist Flex/Ext']
                          ]
        skill_4_disrcard_trials = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                   [34],
                                   [1, 2, 3, 4, 6, 7, 8, 9, 10, 13, 25],
                                   [37],
                                   [37],
                                   [37],
                                   [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
                                   [6, 7, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 30, 33, 34, 38],
                                   [1, 2, 3, 4, 7, 11, 13, 16, 17, 18, 19, 20, 22, 24, 26, 27, 28, 29, 30, 31, 33, 34, 36,
                                    37,
                                    38, 39],
                                   [10, 12, 17, 21, 22, 29, 38, 39],
                                   [2, 11],
                                   [37],
                                   [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                    27,
                                    28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39]
                                   ]
        discard_joints = skill_4_joints
        disrcard_trial = skill_4_disrcard_trials
        outdir = "output" + "Age-Uke" + "/"
        APP_FOLDER += 'AGE UKE/'
    return discard_joints, disrcard_trial, outdir, APP_FOLDER # , all_joints

def readAngle(file, column, window_size, draw,
              discard_joints, disrcard_trial, outdir, APP_FOLDER, skill
              ):
    print('\nfile + outdir:', file+outdir, '         APP_FOLDER: \n', APP_FOLDER, " \n")
    totalDir = 0
    reads = []

    dirs1 = []
    df_temp = pd.DataFrame()
    df = pd.DataFrame()

    col_counter = 0
    first = True

    folders = []
    discarded = []

    for base, dirs, files in os.walk(APP_FOLDER):
        if first == True:
            first = False
            continue
        dirs1.append(str(base))

        discard_counter = 0
        found = False
        for k in discard_joints:
            if file == k[0] and column == k[1]:
                last_folder = str(base).split('/')[-1]
                for r in disrcard_trial[discard_counter]:
                    if last_folder == str(r) != -1:
                        found = True
                        break
                if found == True:
                    break
            discard_counter += 1

        if found == True:
            discarded.append(str(base).split('/')[-1])
            continue

        folders.append(str(base).split('/')[-1])

        path = str(base) + "/" + file
        new_path = str(base) + "/new-" + file

        with open(path, 'r') as f:
            with open(new_path, 'w') as f1:
                next(f)  # skip header line
                for line in f:
                    f1.write(line)
            f1.close()
        f.close()

        df_temp = pd.read_csv(new_path)
        reads.append(df_temp[str(column)].tolist())
        df[col_counter] = df_temp[str(column)]
        col_counter = col_counter + 1

    print("Discarded: ", len(discarded), "     Processed: ", len(folders))

    positions = []
    for x1 in range(0, col_counter):
        arr = list()
        inv_arr = list()
        for v in reads[x1]:
            arr.append(v)
            inv_arr.append(-1 * v)

        data_x = np.arange(start=1, stop=len(arr) + 1, step=1, dtype='int')

        widths = np.arange(1, window_size)  # Widths range should cover the expected width of peaks of interest.

        # peak_indexes = signal.find_peaks_cwt(arr, widths, max_distances = widths * 3)

        # 1

        a = time()
        peak_indexes = signal.find_peaks_cwt(arr, widths)
        write_('CWT_peaks -> ' + file + ' ' + column + ', ' + str(skill) + ' , ' + str(time() - a))
        # 2
        #         peaks = scipy.signal.argrelextrema( np.array(arr), comparator=np.greater,order=5)
        #         peak_indexes = peaks[0]

        # 3
        # peak_indexes,_ = scipy.signal.find_peaks(arr, height=7, distance=2.1)

        # 4
        # peak_indexes = peakutils.peak.indexes(np.array(arr), min_dist=4)

        max_value = max(arr)
        max_index = arr.index(max_value)
        if not (max_index in peak_indexes):
            np.append(peak_indexes, max_index)
            # print("max added")

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
        write_( 'CWT_valley -> ' + file+' '+column +', ' + str(skill) + ' , '+ str(time()-a ) )

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
            col_updated = column.replace("/", "-")

            path_out = outdir + file + "/" + col_updated + "/"
            if not (os.path.exists(path_out)):
                os.makedirs(path_out)
            plt.savefig(path_out + file_name)
            plt.clf()

    return positions, reads

def create_time_pos(skill):
    index1  = -1
    draw = False
    all_time_pos = []
    files, cols, all_joints = get_files_cols_all_joints()
    discard_joints, disrcard_trial, outdir, APP_FOLDER = get_skill_data(skill=skill)   

    for file in files:
        index1 = index1 + 1
        for col in cols[index1]:
            print (file + "  "  + col)
            positions, df = readAngle(file, col, 9, draw, discard_joints, disrcard_trial, outdir, APP_FOLDER, skill)
            time_positions = []

            for i in range(0,len(df)):
                for j in positions[2*i]:
                    time_positions.append( j / len(df[i]))

                for j in positions[2*i+1]:
                    time_positions.append( j / len(df[i]))


            time_positions = sorted(time_positions, key = lambda x:float(x))
            all_time_pos.extend(time_positions)

    all_time_pos = sorted(all_time_pos, key = lambda x:float(x))
    all_time_pos = [ '%.2f' % elem for elem in all_time_pos ]
    all_time_pos = list( dict.fromkeys(all_time_pos) )


    with open('skill_%d model/time-positions.csv'%skill, 'w') as f:
        write = csv.writer(f)
        write.writerow(all_time_pos)

    return files, cols, all_joints, discard_joints, disrcard_trial, outdir, APP_FOLDER


def create_model(skill, draw = True, profiling=False):

    if profiling:
        a = time()

    files, cols, all_joints, discard_joints, disrcard_trial, outdir, APP_FOLDER  = create_time_pos(skill)

    if profiling:
        write_('create_time_pos -> create_model, ' + str(skill) + ', ' + str(time() - a))

    all_min = []
    all_max = []
    all_std = []
    all_mean = []

    index1 = -1

    all_time_pos = []

    for file in files:

        index1 = index1 + 1
        for col in cols[index1]:
            min1 = []
            max1 = []
            std1 = []
            mean1 = []

            positions, df = readAngle(file, col, 9, draw, discard_joints, disrcard_trial, outdir, APP_FOLDER, skill)
            number_of_files = len(df)
            final = []

            temp_list = []
            for i in range(0, number_of_files):
                temp_list.append(df[i][0])

            min1.append(min(temp_list))
            max1.append(max(temp_list))
            mean1.append(np.mean(temp_list))
            std1.append(np.std(temp_list))

            ####################
            time_positions = []
            with open('skill_%d model/time-positions.csv'%skill, newline='') as f:
                reader = csv.reader(f)
                time_positions = list(reader)
            time_positions = time_positions[0]
            time_positions = [float(x) for x in time_positions]

            ####################
            temp_list = []
            for j in time_positions:
                temp_list = []

                for i in range(0, number_of_files):
                    index = int(j * len(df[i]))
                    temp_list.append(df[i][index])

                min1.append(min(temp_list))
                max1.append(max(temp_list))
                mean1.append(np.mean(temp_list))
                std1.append(np.std(temp_list))

            temp_list = []
            for i in range(0, number_of_files):
                temp_list.append(df[i][len(df[i]) - 1])

            min1.append(min(temp_list))
            max1.append(max(temp_list))
            mean1.append(np.mean(temp_list))
            std1.append(np.std(temp_list))

            all_min.append(min1)
            all_max.append(max1)
            all_std.append(std1)
            all_mean.append(mean1)

            stdP = []
            stdM = []
            for i in range(0, len(std1)):
                stdP.append(mean1[i] + std1[i])
                stdM.append(mean1[i] - std1[i])

            final = mean1
            data_x = np.arange(start=1, stop=len(final) + 1, step=1, dtype='int')
            if draw == True:
                (fig, ax) = plt.subplots()
                ax.plot(data_x, final, linestyle='dashed', color='blue', label="mean")

                ax.plot(data_x, max1, linestyle='dashed', color='green', label="max")
                ax.plot(data_x, min1, linestyle='dashed', color='black', label="min")
                ax.plot(data_x, stdP, linestyle='dashed', color='orange', label="stdP")
                ax.plot(data_x, stdM, linestyle='dashed', color='red', label="stdM")

                col_updated = col.replace("/", "-")
                path = outdir + file + "/" + col_updated + "/"
                if not (os.path.exists(path)):
                    os.makedirs(path)

                title = "Overall Curve for: " + file + "-" + col_updated
                plt.title(title)
                plt.legend(loc='best')
                file_name = "1-overall.png"
                plt.savefig(path + file_name)

    for i in range(0, len(all_min)):
        all_min[i].pop(0)
        all_max[i].pop(0)
        all_std[i].pop(0)
        all_mean[i].pop(0)

        all_min[i].pop()
        all_max[i].pop()
        all_std[i].pop()
        all_mean[i].pop()

        all_min[i].insert(0, all_joints[i])
        all_max[i].insert(0, all_joints[i])
        all_std[i].insert(0, all_joints[i])
        all_mean[i].insert(0, all_joints[i])

    temp = time_positions
    temp.insert(0, " ")
    all_mean.insert(0, temp)
    all_min.insert(0, temp)
    all_max.insert(0, temp)
    all_std.insert(0, temp)

    skill_model = 'skill_%d model/'%skill
    with open(skill_model+"mean.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_mean)

    with open(skill_model+"min.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_min)

    with open(skill_model+"max.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_max)

    with open(skill_model+"std.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_std)

    if profiling:
        write_('create_model, ' + str(skill) + ', ' + str(time() - a))  # + '\n' + '-'*50 + '\n'*10)
        with open('time.txt', 'r') as myfile:
            a = myfile.read()
        a = a.split('\n')[:-1]
        df = pd.DataFrame(np.array(a).reshape(-1, 1), columns=['a'])
        df['method'], df['skill'], df['time'] = zip(*df['a'].apply(lambda x: x.split(',')))
        df[['method', 'skill', 'time']].to_csv('Profling skill_%d.csv' % skill, index=False)
        os.remove("time.txt")