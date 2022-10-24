import json
import time
import numpy as np
from matplotlib import pyplot as plt
from Utils.Signal_processing_utils import dx_dt
from scipy import signal

# implementation of key-frame extraction method show in: Artist-Friendly Facial Animation Retargeting:
def graph_simplification_original(x: np.array, t: np.array, k=-1):
    def chord_distance(t_x, chord_start_index, chord_end_index, pt_index):
        # find the error between original curve and the shortest point from the chord
        chord_dir = t_x[chord_end_index] - t_x[chord_start_index]
        chord_length = np.linalg.norm(chord_dir)
        pt_dir = t_x[pt_index] - t_x[chord_end_index]
        proj_on_chord_length = np.dot(chord_dir, pt_dir) / chord_length
        proj_on_chord = chord_dir / chord_length * proj_on_chord_length
        distance = np.linalg.norm(pt_dir - proj_on_chord)
        return distance
    def max_chord_distance(t_x, chord_start_index, chord_end_index):
        # find the maximum error between original curve and the chord
        max_val = -1
        max_index = 0
        for i in range(chord_start_index+1, chord_end_index):
            current_distance = chord_distance(t_x, chord_start_index, chord_end_index, i)
            if (current_distance > max_val):
                max_val = current_distance
                max_index = i
        return max_val, max_index
    def max_chord_distance_simplified(t_x, chord_start_index, chord_end_index):
        line = lambda t : (t_x[chord_end_index][1] - t_x[chord_start_index][1]
                           ) * (t_x[t][0] - t_x[chord_start_index][0]) / float(t_x[chord_end_index][0] - t_x[chord_start_index][0]) + t_x[chord_start_index][1]
        max_val = -1
        max_index = 0
        for i in range(chord_start_index + 1, chord_end_index):
            current_distance = abs(line(i) - t_x[i][1])
            if (current_distance > max_val):
                max_val = current_distance
                max_index = i
        return max_val, max_index
    # concat the two array so it's easier to compute chord distance
    t_x = np.vstack((t, x)).transpose()

    # find maximum error between original curve and constructing a chord between every two points
    E2 = np.zeros((x.shape[0], x.shape[0]))
    E2_path = {}
    for i in range(0, x.shape[0]):
        for j in range(i+2, x.shape[0]):
            max_val, max_index = max_chord_distance_simplified(t_x, i, j)
            E2[i, j] = max_val
    Ek = [E2]
    Ek_path = [E2_path]
    if k > 0:
        # iterate k - 2 times since 2 point paths between any pairs of points is trivial
        for i in range(2, k):
            # iterate through all posible points between the second point and second last to see which
            # one to pause at
            Ei = np.zeros((1, x.shape[0]))
            # placing m at the first spot is equivalent to a two point curve
            Ei[0] = E2[0, E2.shape[0]-1]
            # placing m at the last spot is equivalent to just the previous error
            Ei[-1] = Ek[-1][0, -1]
            min_error = np.inf
            for m in range(1, x.shape[0] - 1):
                # one part of the error is the i point path from start to m
                Ei_k_point_path = Ek[-1][0, m]
                # the second part is from M to end
                Ei_E2 = E2[m, -1]
                # max is used to aggregate the two
                Ei_aggregated = max(Ei_k_point_path. Ei_E2)
                Ei[m] = Ei_aggregated





if __name__ == "__main__":
    # =======================================+++++++++++++=======================================
    # ======================================== load data ========================================
    # =======================================+++++++++++++=======================================

    data_path = "C:/Users/evansamaa/Desktop/Staggered_face/data/live_link_data/20221022_MySlate_8/MySlate_8_Evans_Iphone.csv"
    name_to_mesh = {}
    mesh_to_name = {}
    # load model
    with open("C:/Users/evansamaa/Desktop/Staggered_face/models/mesh_number_to_AU_name.json") as f:
        mesh_to_name = json.load(f)
    with open("C:/Users/evansamaa/Desktop/Staggered_face/models/AU_name_to_mesh_number.json") as f:
        name_to_mesh = json.load(f)
    # load data from file
    columns = []
    rotations_columns = []
    data = []
    rotation_data = []
    times = []
    with open(data_path) as f:
        labels = f.readline()
        columns = labels.split(",")
        rotations_columns = columns[-9:-6]
        columns = columns[2:-10]
        data = f.readlines()
    for i in range(0, len(data)):
        frame_time = data[i].split(",")[0]
        frame_time_list = frame_time.split(":")
        frame_hour, frame_minute, frame_second, frame_milisecond = frame_time_list
        frame_hour = float(frame_hour)
        frame_minute = float(frame_minute)
        frame_second = float(frame_second)
        frame_milisecond = float(frame_milisecond)
        frame_time = frame_milisecond / 60 + frame_second + frame_minute * 60 + frame_hour * 3600
        times.append(frame_time)
        rotation_data.append(data[i].split(",")[-9:-6])
        data[i] = data[i].split(",")[2: -10]
    start_time = times[0]
    for i in range(0, len(times)):
        times[i] = times[i] - start_time
    # compute curve using loaded data:
    length = []
    for i in range(0, len(data)):
        for j in range(0, len(data[i])):
            data[i][j] = float(data[i][j])
        for j in range(0, 3):
            rotation_data[i][j] = float(rotation_data[i][j])

    # =======================================+++++++++++++=======================================
    # ======================================= Compute Key_frames =======================================
    # =======================================+++++++++++++=======================================
    baseline = False
    graph_simplification = True

    data = np.array(data)
    rotation_data = np.array(rotation_data)

    # baseline model
    if baseline == True:
        dData_dt = dx_dt(data)
        d2Data_dt2 = dx_dt(dData_dt)

        pos_peaks = signal.find_peaks(d2Data_dt2[:, 0])[0]
        neg_peaks = signal.find_peaks(-d2Data_dt2[:, 0])[0]

        peaks = np.concatenate((neg_peaks, pos_peaks))
        plt.plot(data[:, 0])
        plt.ylabel("weights")
        plt.xlabel("frames")
        plt.plot(peaks, data[peaks, 0], "o")
        plt.show()
    if graph_simplification == True:
        short_data = data[0:100]
        graph_simplification_original(short_data[:, 0], np.arange(0, short_data.shape[0]))




