import numpy as np
from matplotlib import pyplot as plt

# implementation of key-frame extraction method show in: Artist-Friendly Facial Animation Retargeting:
#TODO: Debug this algorithm and implement backtracking
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
        for j in range(i+1, x.shape[0]):
            max_val, max_index = max_chord_distance_simplified(t_x, i, j)
            E2[i, j] = max_val
            try:
                E2_path[i][j] = []
            except:
                E2_path[i] = {j:[]}
    Ek = [E2]
    Ek_path = [E2_path]
    if k > 2:
        # iterate k - 2 times sinces 2 point paths between any pairs of points is trivial
        for i in range(2, k):
            print(Ek_path[-1])
            # iterate through all posible points between the second point and second last to see which
            # one to pause at
            Ei = np.zeros((1, x.shape[0]))
            Ei_path = {}
            # placing m at the first spot is equivalent to a two point curve
            Ei[0] = E2[0, E2.shape[0]-1]
            # placing m at the last spot is equivalent to just the previous error
            # Ei[-1] = Ek[-1][0, -1]
            for n in range(i - 1, x.shape[0]):
                if n == 1:
                    Ei[0, n] = Ek[-1][0, 1]
                    Ei_path[0] = {n: []}
                else:
                    min_error = np.inf
                    min_m = -1
                    for m in range(1, n):
                        # one part of the error is the i point path from start to m
                        Ei_k_point_path = Ek[-1][0, m]
                        # the second part is from M to end
                        Ei_E2 = E2[m, n]
                        # max is used to aggregate the two
                        Ei_aggregated = max(Ei_k_point_path, Ei_E2)
                        if Ei_aggregated <= min_error:
                            min_error = Ei_aggregated
                            min_m = m
                    Ei[0, n] = min_error
                    if not min_m in Ek_path[-1][0][n]:
                        path_so_far = Ek_path[-1][0][n] + [min_m]
                    else:
                        path_so_far = Ek_path[-1][0][n]
                    try:
                        Ei_path[0][n] = path_so_far
                    except:
                        Ei_path[0] = {n: path_so_far}
            Ek.append(Ei)
            Ek_path.append(Ei_path)
        plt.plot(Ek[-1][0])
        plt.show()

# Implementation to fit the piecewise-linear curve approximation to an arbituary signal
def piece_wise_linear_intervals(x, y, E_line=400):
    """
    Used dynamic programming to find the linear segments that approximate the data
    :param x: array for time
    :param y: arrau fpr values
    :param E_line: penalty of adding a new line
    :return:
    """

    def get_key_points(xs, original_curve, intervals, slope):
        x = [xs[intervals[0][0]]]
        y = [original_curve[intervals[0][0]]]
        for i in range(0, len(intervals)):
            x.append(xs[intervals[i][1]])
            y.append(original_curve[intervals[i][1]])
        x = np.array(x)
        y = np.array(y)
        return x, y
    def plot_piece_wise_lienar_intervals(xs, original_curve, intervals, slope):
        x, y = get_key_points(xs, original_curve, intervals, slope)
        plt.plot(x, y, "o")
        plt.plot(xs, original_curve)
        plt.show()
        return x, y
    def compute_cost(a, b, x, y, E_line=0.1):
        # a and b are indexes into array x and y
        # x is the horizontal axis, y is the vertical
        cost = E_line
        slope = (y[b] - y[a]) / (x[b] - x[a])
        y_int = y[b] - slope * x[b]
        real_y = y[a:b]
        vertical_difference = real_y - (y_int + slope * x[a:b])
        vertical_difference = np.linalg.norm(vertical_difference, ord=1)
        cost = cost + vertical_difference
        return cost
    def traverse_solution(back_track):
        queue = [[0, back_track.shape[0] - 1]]
        sol = []
        while len(queue) > 0:
            current = queue.pop(0)
            current_pointer = int(back_track[current[0], current[1]])
            if current_pointer == -1:
                sol.append([current[0], current[1]])
            else:
                queue = [[current[0], current_pointer], [current_pointer, current[1]]] + queue
        return sol
    M = np.zeros((y.shape[0], y.shape[0]))
    back_track = np.zeros((y.shape[0], y.shape[0]))
    for i in range(1, y.shape[0]):
        # for each diagonal
        diagona_size = y.shape[0] - i
        diag_i = np.zeros((diagona_size,))
        for a in range(0, diagona_size):
            # for each element in the diagonal
            diag_i[a] = compute_cost(a, i + a, x, y, E_line)
            back_track[a, a + i] = -1
            # iterate through the precomputed matrix
            for k in range(1, i):
                combined_cost = M[a, a + i - k] + M[a + (i - k), a + i]
                if combined_cost < diag_i[a]:
                    diag_i[a] = combined_cost
                    back_track[a, a + i] = a + i - k
        M = M + np.diag(diag_i, i)

    # M is the value matrix, backtrack is the matrix that contains the solution
    return traverse_solution(back_track)
def efficient_piece_wise_linear_intervals(x, y):
    '''
    This is a more efficient way of doing it. breaking the signal into chucks of 200 ms. This way instead of O(N^2),
    it's O(40000k) such that k = N/400
    :param x: time of the signal
    :param y: value of the signal
    :return: a list of slopes and intervals [(t0_i, t1_i)]
    '''
    def get_key_points(xs, original_curve, intervals, slope):
        x = [xs[intervals[0][0]]]
        y = [original_curve[intervals[0][0]]]
        for i in range(0, len(intervals)):
            x.append(xs[intervals[i][1]])
            y.append(original_curve[intervals[i][1]])
        x = np.array(x)
        y = np.array(y)
        return x, y
    def plot_piece_wise_lienar_intervals(xs, original_curve, intervals, slope):
        x, y = get_key_points(xs, original_curve, intervals, slope)
        plt.plot(x, y, "o")
        plt.plot(xs, original_curve)
        plt.show()
        return x, y
    def compute_cost(a, b, x, y, E_line=0.1):
        # a and b are indexes into array x and y
        # x is the horizontal axis, y is the vertical
        cost = E_line
        slope = (y[b] - y[a]) / (x[b] - x[a])
        y_int = y[b] - slope * x[b]
        real_y = y[a:b]
        vertical_difference = real_y - (y_int + slope * x[a:b])
        vertical_difference = np.linalg.norm(vertical_difference, ord=1)
        cost = cost + vertical_difference
        return cost
    def traverse_solution(back_track):
        queue = [[0, back_track.shape[0] - 1]]
        sol = []
        while len(queue) > 0:
            current = queue.pop(0)
            current_pointer = int(back_track[current[0], current[1]])
            if current_pointer == -1:
                sol.append([current[0], current[1]])
            else:
                queue = [[current[0], current_pointer], [current_pointer, current[1]]] + queue
        return sol
    L = 200
    # compute E_line
    E_line = (y.max() - y.min()) / 2


    # divide the input to shorter subarrays
    sub_x_lists = []
    sub_y_lists = []
    for i in range(0, int(np.ceil(x.shape[0] / L))):
        sub_x_lists.append(x[int(i * L):int(min((i + 1) * L, x.shape[0]))])
        sub_y_lists.append(y[int(i * L):int(min((i + 1) * L, x.shape[0]))])

    # use dynamic programming to get linear intervals
    pitch_intervals_index = []
    for i in range(0, len(sub_x_lists)):
        sol_index = piece_wise_linear_intervals(sub_x_lists[i], sub_y_lists[i], E_line)
        sol_index = [[val[0] + int(i * L), val[1] + int(i * L)] for val in sol_index]
        sol = [[x[val[0]], x[val[1]]] for val in sol_index]
        pitch_intervals_index = pitch_intervals_index + sol_index
    for i in range(0, len(pitch_intervals_index) - 1):
        if pitch_intervals_index[i][1] < pitch_intervals_index[i + 1][0]:
            pitch_intervals_index[i][1] = pitch_intervals_index[i + 1][0]

    # obtain the slope of these intervals
    pitch_intervals_slopes = []
    for i in range(0, len(pitch_intervals_index)):
        interval_i_index = pitch_intervals_index[i]
        slope = (y[interval_i_index[1]] - y[interval_i_index[0]]) / (x[interval_i_index[1]] - x[interval_i_index[0]])
        pitch_intervals_slopes.append(slope)

    if len(pitch_intervals_slopes) == 1:
        return pitch_intervals_slopes, pitch_intervals_index
    # merge nearby intervals
    pitch_intervals = []
    pitch_slope = []
    prev_begin = 0
    prev_slope = pitch_intervals_slopes[0]
    counting = 1
    for i in range(1, len(pitch_intervals_slopes)):
        current_slope = pitch_intervals_slopes[i]
        if abs(current_slope - prev_slope) <= 30:
            prev_slope = ((prev_slope * counting) + current_slope) / (counting + 1)
            counting = counting + 1
            if i == len(pitch_intervals_slopes) - 1:
                pitch_intervals.append([prev_begin, pitch_intervals_index[i][1]])
                pitch_slope.append(prev_slope)
        else:
            if counting > 1:
                pitch_intervals.append([prev_begin, pitch_intervals_index[i - 1][1]])
                pitch_slope.append(prev_slope)
            else:
                pitch_intervals.append([pitch_intervals_index[i - 1][0], pitch_intervals_index[i - 1][1]])
                pitch_slope.append(prev_slope)
            prev_slope = current_slope
            counting = 1
            prev_begin = pitch_intervals_index[i][0]
            if i == len(pitch_intervals_slopes) - 1:
                pitch_intervals.append([prev_begin, pitch_intervals_index[i][1]])
                pitch_slope.append(prev_slope)
    # pitch_slope = [0 if abs(val) < 25 else val for val in pitch_slope]
    return pitch_slope, pitch_intervals

