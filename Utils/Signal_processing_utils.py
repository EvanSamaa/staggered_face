import numpy as np
from matplotlib import pyplot as plt
from scipy.signal.windows import kaiser
from scipy.interpolate import interp1d
import parselmouth
import librosa

def dx_dt(x: np.array, dt: float = 1, method=1):
    """
    This functio compute first derivative for the input function x using either central or forward differences

    :param x: input array to compute derivative, should be of shape [num of timestamp, num of attributes]
    :param dt: time stamp size
    :param method: method of computing derivative. 1 is forward difference, 2 is central differences
    :return: dx/dt, would be the same size as x. The first and last element are zero.
    """
    out_dx_dt = np.zeros(x.shape)
    if len(x.shape) == 2:
        for j in range(0, x.shape[1]):
            if method == 1:
                for i in range(0, x.shape[0] - 1):
                    out_dx_dt[i, j] = (x[i + 1, j] - x[i, j])/dt
                out_dx_dt[-1, j] = out_dx_dt[-2, j]
            if method == 2:
                for i in range(1, x.shape[0] - 1):
                    out_dx_dt[i, j] = (x[i + 1, j] - x[i - 1, j]) / 2 / dt
                out_dx_dt[-1, j] = out_dx_dt[-2, j]
                out_dx_dt[0, j] = out_dx_dt[1, j]
    elif len(x.shape) == 1:
        if method == 1:
            for i in range(0, x.shape[0] - 1):
                out_dx_dt[i] = (x[i + 1] - x[i]) / dt
            out_dx_dt[-1] = 0
        if method == 2:
            for i in range(1, x.shape[0] - 1):
                out_dx_dt[i] = (x[i + 1] - x[i - 1]) / 2 / dt
            out_dx_dt[-1] = 0
            out_dx_dt[0] = 0
    return out_dx_dt
def intensity_from_signal(x, win_size=441, unit="db"):
    """
    get the intensity of a signal, as implemented in praat
    :param x: input signal, should be a numpy array
    :param win_size: should be an integer, window size for the output signal, default for 441
    :return: the intensity of the signal, in the logarithmic scale (dB)
    """
    total_time_steps = np.ceil(x.shape[0] / win_size)
    # kaiser_window = np.array([0.000356294, 0.000434478, 0.000520568, 0.000615029, 0.000718343, 0.000831007, 0.000953531, 0.00108645, 0.00123029, 0.00138563, 0.00155304, 0.0017331, 0.00192643, 0.00213364, 0.00235538, 0.0025923, 0.00284507, 0.00311438, 0.00340092, 0.00370541, 0.00402859, 0.0043712, 0.00473401, 0.00511779, 0.00552334, 0.00595146, 0.00640297, 0.0068787, 0.00737952, 0.00790627, 0.00845984, 0.00904112, 0.009651, 0.0102904, 0.0109602, 0.0116615, 0.012395, 0.0131618, 0.0139629, 0.0147993, 0.0156718, 0.0165815, 0.0175295, 0.0185167, 0.0195442, 0.0206129, 0.0217239, 0.0228783, 0.024077, 0.0253211, 0.0266117, 0.0279498, 0.0293365, 0.0307728, 0.0322598, 0.0337984, 0.0353898, 0.037035, 0.0387351, 0.0404911, 0.0423039, 0.0441748, 0.0461045, 0.0480944, 0.0501451, 0.0522579, 0.0544336, 0.0566733, 0.0589778, 0.0613483, 0.0637855, 0.0662906, 0.0688641, 0.0715072, 0.0742207, 0.0770055, 0.0798622, 0.082792, 0.0857952, 0.088873, 0.092026, 0.0952549, 0.0985603, 0.101943, 0.105404, 0.108943, 0.112561, 0.116259, 0.120037, 0.123895, 0.127835, 0.131856, 0.13596, 0.140145, 0.144413, 0.148764, 0.153198, 0.157715, 0.162315, 0.166999, 0.171767, 0.176619, 0.181554, 0.186572, 0.191674, 0.19686, 0.202128, 0.20748, 0.212914, 0.218431, 0.224029, 0.229709, 0.23547, 0.241311, 0.247232, 0.253233, 0.259312, 0.26547, 0.271704, 0.278015, 0.284401, 0.290861, 0.297396, 0.304002, 0.31068, 0.317429, 0.324246, 0.331131, 0.338083, 0.3451, 0.352181, 0.359324, 0.366529, 0.373792, 0.381113, 0.388491, 0.395923, 0.403407, 0.410943, 0.418527, 0.426159, 0.433836, 0.441557, 0.449318, 0.457119, 0.464957, 0.47283, 0.480735, 0.488672, 0.496636, 0.504626, 0.512641, 0.520675, 0.52873, 0.5368, 0.544884, 0.55298, 0.561083, 0.569194, 0.577308, 0.585423, 0.593536, 0.601645, 0.609746, 0.617838, 0.625917, 0.633981, 0.642026, 0.650051, 0.658051, 0.666024, 0.673969, 0.68188, 0.689757, 0.697595, 0.705392, 0.713145, 0.720852, 0.728509, 0.736112, 0.743661, 0.751151, 0.75858, 0.765944, 0.773242, 0.780469, 0.787625, 0.794705, 0.801706, 0.808626, 0.815462, 0.822213, 0.828873, 0.835441, 0.841916, 0.848292, 0.854568, 0.860742, 0.866811, 0.872773, 0.878625, 0.884364, 0.889988, 0.895494, 0.900881, 0.906147, 0.911288, 0.916303, 0.92119, 0.925946, 0.93057, 0.935058, 0.939412, 0.943627, 0.947701, 0.951633, 0.955423, 0.959067, 0.962564, 0.965913, 0.969111, 0.97216, 0.975055, 0.977796, 0.980384, 0.982814, 0.985087, 0.987203, 0.989158, 0.990955, 0.99259, 0.994064, 0.995376, 0.996525, 0.99751, 0.998333, 0.998991, 0.999486, 0.999814, 0.999979, 0.999979, 0.999814, 0.999486, 0.998991, 0.998333, 0.99751, 0.996525, 0.995376, 0.994064, 0.99259, 0.990955, 0.989158, 0.987203, 0.985087, 0.982814, 0.980384, 0.977796, 0.975055, 0.97216, 0.969111, 0.965913, 0.962564, 0.959067, 0.955423, 0.951633, 0.947701, 0.943627, 0.939412, 0.935058, 0.93057, 0.925946, 0.92119, 0.916303, 0.911288, 0.906147, 0.900881, 0.895494, 0.889988, 0.884364, 0.878625, 0.872773, 0.866811, 0.860742, 0.854568, 0.848292, 0.841916, 0.835441, 0.828873, 0.822213, 0.815462, 0.808626, 0.801706, 0.794705, 0.787625, 0.780469, 0.773242, 0.765944, 0.75858, 0.751151, 0.743661, 0.736112, 0.728509, 0.720852, 0.713145, 0.705392, 0.697595, 0.689757, 0.68188, 0.673969, 0.666024, 0.658051, 0.650051, 0.642026, 0.633981, 0.625917, 0.617838, 0.609746, 0.601645, 0.593536, 0.585423, 0.577308, 0.569194, 0.561083, 0.55298, 0.544884, 0.5368, 0.52873, 0.520675, 0.512641, 0.504626, 0.496636, 0.488672, 0.480735, 0.47283, 0.464957, 0.457119, 0.449318, 0.441557, 0.433836, 0.426159, 0.418527, 0.410943, 0.403407, 0.395923, 0.388491, 0.381113, 0.373792, 0.366529, 0.359324, 0.352181, 0.3451, 0.338083, 0.331131, 0.324246, 0.317429, 0.31068, 0.304002, 0.297396, 0.290861, 0.284401, 0.278015, 0.271704, 0.26547, 0.259312, 0.253233, 0.247232, 0.241311, 0.23547, 0.229709, 0.224029, 0.218431, 0.212914, 0.20748, 0.202128, 0.19686, 0.191674, 0.186572, 0.181554, 0.176619, 0.171767, 0.166999, 0.162315, 0.157715, 0.153198, 0.148764, 0.144413, 0.140145, 0.13596, 0.131856, 0.127835, 0.123895, 0.120037, 0.116259, 0.112561, 0.108943, 0.105404, 0.101943, 0.0985603, 0.0952549, 0.092026, 0.088873, 0.0857952, 0.082792, 0.0798622, 0.0770055, 0.0742207, 0.0715072, 0.0688641, 0.0662906, 0.0637855, 0.0613483, 0.0589778, 0.0566733, 0.0544336, 0.0522579, 0.0501451, 0.0480944, 0.0461045, 0.0441748, 0.0423039, 0.0404911, 0.0387351, 0.037035, 0.0353898, 0.0337984, 0.0322598, 0.0307728, 0.0293365, 0.0279498, 0.0266117, 0.0253211, 0.024077, 0.0228783, 0.0217239, 0.0206129, 0.0195442, 0.0185167, 0.0175295, 0.0165815, 0.0156718, 0.0147993, 0.0139629, 0.0131618, 0.012395, 0.0116615, 0.0109602, 0.0102904, 0.009651, 0.00904112, 0.00845984, 0.00790627, 0.00737952, 0.0068787, 0.00640297, 0.00595146, 0.00552334, 0.00511779, 0.00473401, 0.0043712, 0.00402859, 0.00370541, 0.00340092, 0.00311438, 0.00284507, 0.0025923, 0.00235538, 0.00213364, 0.00192643, 0.0017331, 0.00155304, 0.00138563, 0.00123029, 0.00108645, 0.000953531, 0.000831007, 0.000718343, 0.000615029, 0.000520568, 0.000434478, 0.000356294])
    kaiser_window = kaiser(win_size, 5)
    intensity = []
    for i in range(0, int(total_time_steps)):
        end_up = min((i+1)*win_size, x.shape[0])
        current_kaiser_window = kaiser_window[0:(end_up - i*win_size)]
        current = current_kaiser_window * x[i*win_size:end_up] * x[i*win_size:end_up]
        current_frame_intensity = current.sum()/kaiser_window.sum()/4.0e-10
        # current_frame_intensity = current.sum()/kaiser_window.sum()
        if current_frame_intensity <= 1.0e-30:
            current_frame_intensity = 0
        else:
            # if in the absolute value scale
            if unit == "pow":
                current_frame_intensity = np.sqrt(current_frame_intensity)
            # in the case where logarithmic scale should be used
            elif unit == "db":
                current_frame_intensity = 10*np.log10(current_frame_intensity)
        intensity.append(current_frame_intensity)
    return np.array(intensity)
def pitch_from_signal(x, win_size=441):
    """
    get the fundamental frequency of the signal, as per done in praat
    by default it assumes the signal has a sample size of 44100 Hz, and
    computes 100 times per second
    :param x: input signal
    :return: a signal of intensity. Should have ceil(sample_size / 441) samples
    """
    snd = parselmouth.Sound(x)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    xs = pitch.xs()
    out_interp = interp1d(xs, pitch_values, fill_value="extrapolate")
    out = np.arange(0, np.ceil(x.shape[0] / win_size))
    return out_interp(out)
def interpolate1D(arr_t, arr_x, t):
    arr_t = np.array(arr_t)
    if t < arr_t[0]:
        return arr_x[0]
    elif t >= arr_t[-1]:
        return arr_x[-1]
    else:
        for i in range(0, arr_t.shape[0] - 1):
            if arr_t[i] <= t and arr_t[i + 1] > t:
                return arr_x[i]
    print("Error")
def gen_gaussian_window(half_win_size, stdev):
    out = np.zeros((half_win_size * 2 + 1, ))
    M = float(half_win_size)
    for i in range(0, int(half_win_size * 2 + 1)):
        out[i] = float(np.exp(-0.5 * np.power((i - M)/stdev/M, 2)))
    return out

def sparse_key_smoothing(arr_t, arr_x, fps=24,
                         smoothing_win_size = 1):
    new_temp_x = arr_x.copy()
    new_temp_2 = arr_x.copy()
    if len(new_temp_2) > 2:
        for t in range(1, len(new_temp_2)-1):
            time_to_next = arr_t[t+1] - arr_t[t]
            time_to_prev = arr_t[t] - arr_t[t-1]
            t_nearest = np.minimum(time_to_next, time_to_prev)
            if (t_nearest < smoothing_win_size/fps):
                segment_size = np.maximum(t_nearest, 0.2 / fps)
                actual_half_window_size = np.round(smoothing_win_size / fps / segment_size)
                actual_half_window_size = np.maximum(actual_half_window_size, 1)
                segment_size = smoothing_win_size / actual_half_window_size / fps
                # window = np.ones((int(actual_half_window_size) * 2 + 1, ))
                window = gen_gaussian_window(int(actual_half_window_size), 0.3).tolist()
                vals = 0
                count = 0
                for w in range(0, int(actual_half_window_size) * 2 + 1):
                    w_normalized = w - actual_half_window_size
                    interpolated_val = interpolate1D(arr_t, arr_x, arr_t[t] + w_normalized * segment_size)
                    vals += window[w] * interpolated_val
                    count += window[w]
                new_temp_x[t] = 0.8 * (vals/count) + 0.2 * (new_temp_2[t])
    try:
        new_temp_x = new_temp_x.tolist()
    except:
        pass
    return new_temp_x

def laplacian_smoothing(arr, iteration=1):
    try:
        shape = arr.shape
        for i in range(iteration):
            out = np.zeros(arr.shape)
            out[0] = arr[0]
            out[-1] = arr[-1]
            for j in (1, shape[0]-1):
                out[j] = (arr[j-1] + arr[j] + arr[j+1])/3.0
            arr = out
        return out
    except:
        for i in range(iteration):
            out = []
            out.append(arr[0])
            for j in range(1, len(arr)-1):
                out.append((arr[j-1] + arr[j] + arr[j+1])/3.0)
            out.append(arr[-1])
            arr = out.copy()
        return out
if __name__ == "__main__":
    file_path = "F:/MASC/JALI_neck/data/neck_rotation_values/CNN/"
    file_name = "cnn_borderOneGuy.wav"
    audio, sr = librosa.load(file_path + file_name, sr=44100)
    intensity = intensity_from_signal(audio)
    pitch = pitch_from_signal(audio)
    print(intensity.shape, pitch.shape)