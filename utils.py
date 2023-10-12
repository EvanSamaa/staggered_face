import copy
import numpy as np
import math
import open3d as o3d
import json, time, moviepy, os
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
class CubicSplineInterpolation:
    # this class is limited to only be able to fit coninious curves (for example the discontinious)
    # jk i was wrong I'm dumb lol
    def __init__(self, x, y):
        self.kind = "cubic spline"
        self.control_points: np.array = None # in the shape of [T, 2]
        self.x = x
        self.y = y
        # find all the k by solving Ak = b
        A = np.zeros([self.x.shape[0], self.x.shape[0]])
        b = np.zeros(self.x.shape)
        # find A
        for i in range(1, self.x.shape[0]-1):
            A[i, i-1] = 1.0 / (self.x[i] - self.x[i-1])
            A[i, i] = 2 * (1.0 / (self.x[i] - self.x[i-1]) + 1.0 / (self.x[i+1] - self.x[i]))
            A[i, i+1] = 1.0 / (self.x[i+1] - self.x[i])
            b[i] = 3 * ((self.y[i] - self.y[i-1])/(self.x[i] - self.x[i-1])**2 + (self.y[i+1] - self.y[i])/(self.x[i+1] - self.x[i])**2)
        A[0, 0] = 2 / (self.x[1] - self.x[0])
        A[0, 1] = 1 / (self.x[1] - self.x[0])
        A[-1, -2] = 1 / (self.x[-1] - self.x[-2])
        A[-1, -1] = 2 / (self.x[-1] - self.x[-2])
        # find b
        b[0] = 3 * ((self.y[1]-self.y[0])/(self.x[1]-self.x[0])**2)
        b[-1] = 3 * ((self.y[-1]-self.y[-2])/(self.x[-1]-self.x[-2])**2)
        # solve for k
        self.k = np.linalg.solve(A, b)
        return
    def binary_search(self, t):
        if t <= self.x[0]:
            return 0
        elif t > self.x[-1]:
            return self.x.shape[0]-1
        # find it
        left, right = 0, self.x.shape[0] - 1
        index = -1
        
        while left <= right:
            mid = (left + right) // 2
            
            if self.x[mid] < t:
                index = mid
                left = mid + 1
            else:
                right = mid - 1
        return index
    def eval_single(self, x):
        if x <= self.x[0]:
            return self.y[0]
        elif x >= self.x[-1]:
            return self.y[-1]
        i = self.binary_search(x)
        x1 = self.x[i]
        x2 = self.x[i+1]
        y1 = self.y[i]
        y2 = self.y[i+1]
        k1 = self.k[i]
        k2 = self.k[i+1]
        t = (x - x1) / (x2 - x1)
        a = k1 * (x2 - x1) - (y2 - y1)
        b = -k2 * (x2 - x1) + (y2 - y1)
        y = (1 - t) * y1 + t * y2 + t * (1-t) * ((1-t) * a + t * b)
        return y
    def eval(self, ts):
        out = np.zeros(ts.shape)
        for i in range(0, ts.shape[0]):
            out[i] = self.eval_single(ts[i])
        return out
class CatmullRomSplineInterpolation:
    # this class is limited to only be able to fit coninious curves (for example the discontinious)
    # jk i was wrong I'm dumb lol
    def __init__(self, x, y, tao=0.5, lower_bound=None):
        x = np.array(x)
        y = np.array(y)
        self.lower_bound = lower_bound
        self.kind = "catmull rom"
        self.x = np.zeros([x.shape[0] + 2, ])
        self.x[1:-1] = x
        self.x[0] = x[0]
        self.x[-1] = x[-1]
        self.y = np.zeros([y.shape[0] + 2, ])
        self.y[1:-1] = y
        self.y[0] = y[0]
        self.y[-1] = y[-1]
        self.tension_mat = [[0, 1, 0, 0], 
                            [-tao, 0, tao, 0], 
                            [2 * tao, tao - 3, 3 - 2 * tao, -tao],
                            [-tao, 2 - tao, tao - 2, tao]]
        self.tension_mat = np.array(self.tension_mat)
        return
    def binary_search(self, t):
        if t <= self.x[0]:
            return 0
        elif t > self.x[-1]:
            return self.x.shape[0]-1
        # find it
        left, right = 0, self.x.shape[0] - 1
        index = -1
        
        while left <= right:
            mid = (left + right) // 2
            
            if self.x[mid] < t:
                index = mid
                left = mid + 1
            else:
                right = mid - 1
        return index
    def eval_single(self, x):
        # here we ensure that we will never get to the end of the curve
        if x <= self.x[1]:
            return self.y[0]
        elif x >= self.x[-2]:
            return self.y[-1]
        i = self.binary_search(x)
        p0 = self.y[i-1]
        p1 = self.y[i]
        p2 = self.y[i+1]
        p3 = self.y[i+2]
        t0 = self.x[i]
        t1 = self.x[i+1]
        u = (x - t0) / (t1 - t0)
        u_vec = np.array([1, u, u**2, u**3])
        p_vec = np.array([p0, p1, p2, p3])

        val = u_vec.dot(self.tension_mat @ p_vec)
        if not self.lower_bound is None:
            val = np.maximum(val, self.lower_bound)
        return val
    def eval(self, ts):
        out = np.zeros(ts.shape)
        for i in range(0, ts.shape[0]):
            out[i] = self.eval_single(ts[i])
        return out

def binary_search_for_left(ts, t):
    # boundary conditions
    if t <= ts[0]:
        return 0
    elif t > ts[-1]:
        return ts.shape[0]-1
    # find it
    left, right = 0, ts.shape[0] - 1
    index = -1
    # use recursion
    while left <= right:
        mid = (left + right) // 2
        
        if ts[mid] < t:
            index = mid
            left = mid + 1
        else:
            right = mid - 1
    return index
class BasicBlendshapeModel:
    def __init__(self):
        # for basic functionalities. Using Trimesh as the basis
        self.name_map =  {'eyeblink_r': 'eyeblinkright', 'eyelookdown_r': 'eyelookdownright', 'eyelookin_r': 'eyelookinright', 'eyelookout_r': 'eyelookoutright', 'eyelookup_r': 'eyelookupright', 'eyesquint_r': 'eyesquintright', 'eyewide_r': 'eyewideright', 'eyeblink_l': 'eyeblinkleft', 'eyelookdown_l': 'eyelookdownleft', 'eyelookin_l': 'eyelookinleft', 'eyelookout_l': 'eyelookoutleft', 'eyelookup_l': 'eyelookupleft', 'eyesquint_l': 'eyesquintleft', 'eyewide_l': 'eyewideleft', 'jawforward': 'jawforward', 'jawright': 'jawright', 'jawleft': 'jawleft', 'jawopen': 'jawopen', 'mouthclose': 'mouthclose', 'mouthfunnel': 'mouthfunnel', 'mouthpucker': 'mouthpucker', 'mouthright': 'mouthright', 'mouthleft': 'mouthleft', 'mouthsmile_r': 'mouthsmileright', 'mouthsmile_l': 'mouthsmileleft', 'mouthfrown_r': 'mouthfrownright', 'mouthfrown_l': 'mouthfrownleft', 'mouthdimple_r': 'mouthdimpleright', 'mouthdimple_l': 'mouthdimpleleft', 'mouthstretch_r': 'mouthstretchright', 'mouthstretch_l': 'mouthstretchleft', 'mouthrolllower': 'mouthrolllower', 'mouthrollupper': 'mouthrollupper', 'mouthshruglower': 'mouthshruglower', 'mouthshrugupper': 'mouthshrugupper', 'mouthpress_r': 'mouthpressright', 'mouthpress_l': 'mouthpressleft', 'mouthlowerdown_r': 'mouthlowerdownright', 'mouthlowerdown_l': 'mouthlowerdownleft', 'mouthupperup_r': 'mouthupperupright', 'mouthupperup_l': 'mouthupperupleft', 'browdown_r': 'browdownright', 'browdown_l': 'browdownleft', 'browinnerup': 'browinnerup', 'browouterup_r': 'browouterupright', 'browouterup_l': 'browouterupleft', 'cheekpuff': 'cheekpuff', 'cheeksquint_r': 'cheeksquintright', 'cheeksquint_l': 'cheeksquintleft', 'nosesneer_r': 'nosesneerright', 'nosesneer_l': 'nosesneerleft'}
        self.neutral_mesh : o3d.geometry = None
        self.blendshape_mesh : dict[str, o3d.geometry] = {}
        self.weight : dict[str, float] = {}
        self.translation: np.array = np.zeros([3, ])
        self.scale_factor: float = 1
        self.visualization_mesh: o3d.geometry = None
    def translate(self, delta_pos):
        self.translation += delta_pos
    def scale(self, scaling=1):
        self.scale_factor = scaling
    def eval(self):
        out_vers = np.asarray(self.neutral_mesh.vertices).copy()
        for i in self.weight:
            try:
                shape_i = np.asarray(self.blendshape_mesh[self.name_map[i.lower()]].vertices)
            except:
                shape_i = np.asarray(self.blendshape_mesh[i.lower()].vertices)
            out_vers += shape_i * self.weight[i]
        out_vers = out_vers * self.scale_factor + np.expand_dims(self.translation, axis=0)
        self.visualization_mesh.vertices = o3d.utility.Vector3dVector(out_vers)
        return self.visualization_mesh
    def facing_dire(self):
        facing_dir = np.mean(self.neutral_mesh.face_normals, axis=0)
        facing_dir /= np.linalg.norm(facing_dir)
        return facing_dir
def load_blendshape_model(path: str, model: BasicBlendshapeModel):
    neutral_path = os.path.join(*[path, "Neutral.obj"])
    blendshape_paths = []
    folder_content = os.listdir(path)
    blendshape_name = []
    for b in folder_content:
        if b[-4:] != "gltf" and b[:7] != "Neutral":
            blendshape_paths.append(os.path.join(path, b))
            blendshape_name.append(b.split(".")[0].lower())
    model.neutral_mesh = o3d.io.read_triangle_mesh(neutral_path)
    # model.neutral_mesh.vertices = Rotation.from_euler("xyz", [0, 0, 90], degrees=True).apply(model.neutral_mesh.vertices)
    for i in range(0, len(blendshape_paths)):
        model.blendshape_mesh[blendshape_name[i]] = o3d.io.read_triangle_mesh(blendshape_paths[i])
        neutral_verts = np.asarray(model.neutral_mesh.vertices)
        verts = np.asarray(model.blendshape_mesh[blendshape_name[i]].vertices)
        model.blendshape_mesh[blendshape_name[i]].vertices = o3d.utility.Vector3dVector(verts - neutral_verts)
    model.visualization_mesh = copy.deepcopy(model.neutral_mesh)
    return model
def display_blendshape_model_o3d(model:BasicBlendshapeModel):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Face_visualize', width=800, height=600)
    vis.add_geometry(model.eval())
    vis.run()
    vis.destroy_window()
def maximum_displacement_vertex(model: BasicBlendshapeModel):
    maximum_displacement_vertices = {}
    for key in model.blendshape_mesh:
        mesh = model.blendshape_mesh[key].vertices
        disp = np.linalg.norm(mesh, axis=1)
        maximum_displacement_vertices[key] = mesh[np.argmax(disp)] + model.neutral_mesh.vertices[np.argmax(disp)]
    return maximum_displacement_vertices
def maximum_k_displacement_vertex(model: BasicBlendshapeModel, k=1):
    maximum_displacement_vertices = {}
    maximum_displacement_indices = {}
    indices = set([])
    for key in model.blendshape_mesh:
        mesh = np.asarray(model.blendshape_mesh[key].vertices)
        disp = np.linalg.norm(mesh, axis=1)
        ind = np.argpartition(disp, -k)[-k:].tolist()
        indices.update(ind)
        maximum_displacement_vertices[key] = mesh[ind] + np.asarray(model.neutral_mesh.vertices)[ind]
        maximum_displacement_indices[key] = ind
    return maximum_displacement_vertices, maximum_displacement_indices, list(indices)
class BlendshapeAnimation:
    def __init__(self, model: BasicBlendshapeModel, ar_data_path:str = "null", fps=30, ts = None, values_dict=None):
        self.name_map =  {'eyeblink_r': 'eyeblinkright', 'eyelookdown_r': 'eyelookdownright', 'eyelookin_r': 'eyelookinright', 'eyelookout_r': 'eyelookoutright', 'eyelookup_r': 'eyelookupright', 'eyesquint_r': 'eyesquintright', 'eyewide_r': 'eyewideright', 'eyeblink_l': 'eyeblinkleft', 'eyelookdown_l': 'eyelookdownleft', 'eyelookin_l': 'eyelookinleft', 'eyelookout_l': 'eyelookoutleft', 'eyelookup_l': 'eyelookupleft', 'eyesquint_l': 'eyesquintleft', 'eyewide_l': 'eyewideleft', 'jawforward': 'jawforward', 'jawright': 'jawright', 'jawleft': 'jawleft', 'jawopen': 'jawopen', 'mouthclose': 'mouthclose', 'mouthfunnel': 'mouthfunnel', 'mouthpucker': 'mouthpucker', 'mouthright': 'mouthright', 'mouthleft': 'mouthleft', 'mouthsmile_r': 'mouthsmileright', 'mouthsmile_l': 'mouthsmileleft', 'mouthfrown_r': 'mouthfrownright', 'mouthfrown_l': 'mouthfrownleft', 'mouthdimple_r': 'mouthdimpleright', 'mouthdimple_l': 'mouthdimpleleft', 'mouthstretch_r': 'mouthstretchright', 'mouthstretch_l': 'mouthstretchleft', 'mouthrolllower': 'mouthrolllower', 'mouthrollupper': 'mouthrollupper', 'mouthshruglower': 'mouthshruglower', 'mouthshrugupper': 'mouthshrugupper', 'mouthpress_r': 'mouthpressright', 'mouthpress_l': 'mouthpressleft', 'mouthlowerdown_r': 'mouthlowerdownright', 'mouthlowerdown_l': 'mouthlowerdownleft', 'mouthupperup_r': 'mouthupperupright', 'mouthupperup_l': 'mouthupperupleft', 'browdown_r': 'browdownright', 'browdown_l': 'browdownleft', 'browinnerup': 'browinnerup', 'browouterup_r': 'browouterupright', 'browouterup_l': 'browouterupleft', 'cheekpuff': 'cheekpuff', 'cheeksquint_r': 'cheeksquintright', 'cheeksquint_l': 'cheeksquintleft', 'nosesneer_r': 'nosesneerright', 'nosesneer_l': 'nosesneerleft'}
        self.model: BasicBlendshapeModel = model
        self.weight_over_time: dict[str, CatmullRomSplineInterpolation] = {}
        if ar_data_path != "null":
            self.ts, values_dict = self.load_apple_motion(fps, ar_data_path)
            for key in values_dict:
                self.weight_over_time[key] = CatmullRomSplineInterpolation(self.ts, values_dict[key])
        else:
            # if we are not reading from file, we taking a pre-interpolated value_dict (i.e. curve for each AU is already interpolated in the range of ts)
            self.ts = ts
            for key in values_dict:
                try:
                    self.weight_over_time[self.name_map[key.lower()]] = CatmullRomSplineInterpolation(self.ts, values_dict[key])
                except:
                    self.weight_over_time[key.lower()] = CatmullRomSplineInterpolation(self.ts, values_dict[key])
        self.visualization_mesh: o3d.geometry = model.eval()
    def translate(self, vec):
        # vec is of shape [3, ]
        self.model.translate(vec)
    def eval(self, t):
        out_weight = {}
        for key in self.weight_over_time:
            weight = self.weight_over_time[key].eval_single(t)
            out_weight[key] = weight
        self.model.weight = out_weight
        mesh_t = self.model.eval()
        self.visualization_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_t.vertices))
        return self.visualization_mesh
    def load_apple_motion(self, recordingfps=30, calibration_data_path=""):
        name_to_mesh = {}
        mesh_to_name = {}
        # load model
        with open("/Users/evanpan/Documents/GitHub/staggered_face/models/joonho_mesh_number_to_AU_name.json") as f:
            mesh_to_name = json.load(f)
        with open("/Users/evanpan/Documents/GitHub/staggered_face/models/joonho_AU_name_to_mesh_number.json") as f:
            name_to_mesh = json.load(f)
        # load data from file
        columns = []
        raw_cal_data = []
        cal_data = []
        times = []
        with open(calibration_data_path) as f:
            labels = f.readline()
            columns = labels.split(",")
            columns = columns[2:]
            raw_cal_data = f.readlines()
        for i in range(0, len(raw_cal_data)):
            frame_time = raw_cal_data[i].split(",")[0]
            frame_time_list = frame_time.split(":")
            frame_hour, frame_minute, frame_second, frame_frame = frame_time_list
            frame_hour = float(frame_hour)
            frame_minute = float(frame_minute)
            frame_second = float(frame_second)
            frame_frame = float(frame_frame)
            frame_time = frame_frame/recordingfps + frame_second + frame_minute * 60 + frame_hour * 3600        
            frame_cal_data = raw_cal_data[i].split(",")
            if len(frame_cal_data) > 20:
                times.append(frame_time)
                numerical_values = [float(x) for x in frame_cal_data[2:]]
                cal_data.append(numerical_values)
        start_time = times[0]
        for i in range(0, len(times)):
            times[i] = times[i] - start_time
        cal_data = np.array(cal_data)
        values_dict = {}
        for i in range(0, len(columns)-10):
            try:
                values_dict[self.name_map[columns[i].lower()]] = cal_data[:, i]
            except:
                values_dict[columns[i].lower()] = cal_data[:, i]
        times = np.array(times)
        return times, values_dict
def play_animation(animation:BlendshapeAnimation, save_video=False, video_path="just_animation.mp4"):
    # introduce call back functions to stop the animation
    start_t = time.time()
    t = 0
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='Face_visualize', width=1800, height=600)
    vis.get_render_option().mesh_show_wireframe = True
    vis.add_geometry(animation.eval(0))
    # vis.run()
    frames = []
    while t < animation.ts[-1]:
        if save_video:
            dt = 1/30.0
            t = t + dt
        else:
            t = time.time() - start_t
        vis.update_geometry(animation.eval(t))
        event = vis.poll_events()
        vis.update_renderer()
        if save_video:
            im = vis.capture_screen_float_buffer(True)
            frames.append((np.asarray(im)))
    if save_video:
        # if this does work, pip install upgrade "moviepy" lol
        # Define the video writer
        uint8_frames = [(f*255).astype(np.uint8) for f in frames]
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(uint8_frames, fps=30)
        clip.write_videofile(video_path)
    vis.destroy_window()
def load_apple_motion(motion_data_path = "", recordingfps=30, calibration_data_path="", include_headeye=False):
    name_to_mesh = {}
    mesh_to_name = {}
    # load model
    with open("/Users/evanpan/Documents/GitHub/staggered_face/models/joonho_mesh_number_to_AU_name.json") as f:
        mesh_to_name = json.load(f)
    with open("/Users/evanpan/Documents/GitHub/staggered_face/models/joonho_AU_name_to_mesh_number.json") as f:
        name_to_mesh = json.load(f)
    # load data from file
    columns = []
    raw_cal_data = []
    cal_data = []
    times = []
    with open(calibration_data_path) as f:
        labels = f.readline()
        columns = labels.split(",")
        columns = columns[2:]
        raw_cal_data = f.readlines()
    for i in range(0, len(raw_cal_data)):
        frame_time = raw_cal_data[i].split(",")[0]
        frame_time_list = frame_time.split(":")
        frame_hour, frame_minute, frame_second, frame_frame = frame_time_list
        frame_hour = float(frame_hour)
        frame_minute = float(frame_minute)
        frame_second = float(frame_second)
        frame_frame = float(frame_frame)
        frame_time = frame_frame/recordingfps + frame_second + frame_minute * 60 + frame_hour * 3600        
        frame_cal_data = raw_cal_data[i].split(",")
        if len(frame_cal_data) > 20:
            times.append(frame_time)
            numerical_values = [float(x) for x in frame_cal_data[2:]]
            cal_data.append(numerical_values)
    start_time = times[0]
    for i in range(0, len(times)):
        times[i] = times[i] - start_time
    cal_data = np.array(cal_data)
    values_dict = {}
    if include_headeye:
        for i in range(0, len(columns)):
            values_dict[columns[i]] = cal_data[:, i]
    else:
        for i in range(0, len(columns)-10):
            values_dict[columns[i]] = cal_data[:, i]
    return times, values_dict

class SharkFinCurve():
    def __init__(self, all_frames, all_key_frames, ts, curve):
        # all_frames: are all the keyframes in the sharkfin carve
        # curve: is the data that the keyframe points to
        self.all_frame_ids = all_frames.copy()  # this index into keyframes
        self.key_frame_ids = []                 # this index into keyframes
        self.key_frames = all_key_frames        # this index into curve
        self.ts = ts
        self.curve = copy.deepcopy(curve)       # this is the value
        self.fit_sharkfin()
    def get_min_key_frames(self):
        min_key_frames = [self.key_frames[j][0] for j in self.key_frame_ids]
        return min_key_frames
    def get_x(self, kfid):
        return self.key_frames[kfid][0]
    def get_y(self, kfid):
        return self.curve[self.key_frames[kfid][0]]
    def get_peak(self):
        kf0 = self.get_x(self.all_frame_ids[0])
        kf1 = self.get_x(self.all_frame_ids[1])
        peak_x = np.argmax(self.curve[kf0:kf1]) + kf0
        peak_y = self.curve[peak_x]
        return peak_x, peak_y
    def get_range(self):
        kf0 = self.get_x(self.all_frame_ids[0])
        kf1 = self.get_x(self.all_frame_ids[1])
        return [self.ts[kf0], self.ts[kf1]]
    def get_all_key_frames(self):
        all_key_frames = [self.key_frames[j][0] for j in self.all_frame_ids]
        return all_key_frames
    def fit_sharkfin_heuristic(self):
        self.key_frame_ids = [self.all_frame_ids[0], 0, 0, self.all_frame_ids[-1]]
        # define the peak key frame as the one with the maximum ascent
        max_ascend = -1
        max_ascend_id = -1
        for i in range(0, len(self.all_frame_ids)-1):
            ascend = (self.curve[self.key_frames[self.all_frame_ids[i]][0]] - self.curve[self.key_frames[self.all_frame_ids[0]][0]]
                     )/(self.key_frames[self.all_frame_ids[i]][0] - self.key_frames[self.all_frame_ids[0]][0])
            if ascend > max_ascend:
                max_ascend_id = self.all_frame_ids[i]
        max_descend = 1
        max_descend_id = -1
        # define the sustain key frame as the one with the maximum descent ()
        for i in range(0, len(self.all_frame_ids)-1):
            descend = (self.curve[self.key_frames[self.all_frame_ids[-1]][0]] - self.curve[self.key_frames[self.all_frame_ids[i]][0]]
                     )/(self.key_frames[self.all_frame_ids[-1]][0] - self.key_frames[self.all_frame_ids[i]][0])
            if descend < max_descend:
                max_descend_id = self.all_frame_ids[i]
        self.key_frame_ids[1] = max_ascend_id
        self.key_frame_ids[2] = max_descend_id
        # print(self.key_frame_ids)
    def fit_sharkfin_least_square(self):
        # does not work with catmull rom because the parameterization is dependent on the 2 central points, which we are trying to find
        self.key_frame_ids = [self.all_frame_ids[0], 0, 0, self.all_frame_ids[-1]]
        tao = 0.2
        tension_mat = np.array([[0, 1, 0, 0], 
                    [-tao, 0, tao, 0], 
                    [2 * tao, tao - 3, 3 - 2 * tao, -tao],
                    [-tao, 2 - tao, tao - 2, tao]])
        xs = np.arange(self.key_frames[self.key_frame_ids[0]][0], self.key_frames[self.key_frame_ids[-1]][0]+1)
        ys = self.curve[self.key_frames[self.key_frame_ids[0]][0]:self.key_frames[self.key_frame_ids[-1]][0]+1]
        X_mat = np.array([np.ones(xs.shape), xs, np.square(xs), np.power(xs, 3)]).T
        print(xs, ys)
        ys = np.expand_dims(ys, axis=1) - self.key_frames[self.key_frame_ids[0]][0] * X_mat @ tension_mat[:, 0:1] - self.key_frames[self.key_frame_ids[-1]][0] * X_mat @ tension_mat[:, 3:4]
        # solve Ax = b where A = X_mat[1:3], x = [p1, p2], b = y
        A_mat = X_mat[:, 1:3]
        p_solv = np.linalg.inv(A_mat.T @ A_mat) @ A_mat.T @ ys
        # now try to find keyframe that most closely resemble these)
        # for i in range(0, len(self.key_frame_ids)):
    def fit_sharkfin_best_hull(self):
        # try to find the best hull greedily 
        def find_above(points_x, points_y, x, y):
            # this function fins how might higher the line constructed with points_x_y is compare to x,y
            if x >= points_x[1] and points_x[1] != points_x[2]:
                x0 = points_x[1]
                x1 = points_x[2]
                y0 = points_y[1]
                y1 = points_y[2]
            elif x < points_x[1] and points_x[1] != points_x[0]:
                x0 = points_x[0]
                x1 = points_x[1]
                y0 = points_y[0]
                y1 = points_y[1]
            else:
                return 0
            
            val = (x - x0) / (x1 - x0) * (y1 - y0) + y0
            return val - y
        def search_sub_list(start, end):
            st = start   
            ed = end
            distances = []
            for i in range(st, ed+1):
                # construct a hull using the 2 starting point and a point in the middle. 
                total_above_ness = 0
                for j in range(st, ed+1):
                    total_above_ness += find_above([self.get_x(self.all_frame_ids[st]), self.get_x(self.all_frame_ids[i]), self.get_x(self.all_frame_ids[ed])], 
                                                [self.get_y(self.all_frame_ids[st]), self.get_y(self.all_frame_ids[i]), self.get_y(self.all_frame_ids[ed])], 
                                                self.get_x(self.all_frame_ids[j]), self.get_y(self.all_frame_ids[j]))
                distances.append(total_above_ness)
            return np.argmax(distances) + start
        # print(distances)
        self.key_frame_ids = [self.all_frame_ids[0], 0, 0, self.all_frame_ids[-1]]
        if len(self.all_frame_ids) <= 4: 
            self.key_frame_ids = self.all_frame_ids.copy()
            return
        # find a starting point for search
        distances = 0
        distances = [-1000]
        st = 0
        ed = len(self.all_frame_ids)-1
        for i in range(st+1, ed):
            # construct a hull using the 2 starting point and a point in the middle. 
            total_above_ness = 0
            for j in range(st+1, ed):
                total_above_ness += find_above([self.get_x(self.all_frame_ids[st]), self.get_x(self.all_frame_ids[i]), self.get_x(self.all_frame_ids[ed])], 
                                               [self.get_y(self.all_frame_ids[st]), self.get_y(self.all_frame_ids[i]), self.get_y(self.all_frame_ids[ed])], 
                                               self.get_x(self.all_frame_ids[j]), self.get_y(self.all_frame_ids[j]))
            distances.append(total_above_ness)
        distances.append(-1000)
        # seek the left and right side of the starting point to find more points
        p_mid = np.argmax(distances)
        p1 = -1
        p2 = -1
        if p_mid > st:
            p1 = search_sub_list(st, p_mid)
        else:
            p1 = p_mid
        if p_mid < ed - 1:
            p2 = search_sub_list(p_mid, ed)
        else:
            p2 = p_mid
            
        self.key_frame_ids[1] = self.all_frame_ids[p1]
        self.key_frame_ids[2] = self.all_frame_ids[p2]
    def fit_sharkfin(self):
        self.fit_sharkfin_best_hull()        
class SuperFrame():
    def __init__(self, groupings):
        # Here groupings are assumed to be 
        # these are all gonna be in term of frames
        # groupings {}
        grouping_AUs = [x[0] for x in groupings]
        grouping_frames = [x[1] for x in groupings]
        self.frame_center = math.floor(np.mean(grouping_frames))
        self.frame_off_sets = {}
        for i in range(0, len(grouping_AUs)):
            self.frame_off_sets[grouping_AUs[i]] = grouping_frames[i] - self.frame_center
    def set_super_frame(self, model:BasicBlendshapeModel, vals_dict):
        for key in vals_dict:
            model.weight[key] = vals_dict[key][self.frame_center]
            # model.weight[key] = 0
        for key in self.frame_off_sets:
            model.weight[key] = np.maximum(vals_dict[key][self.frame_center + self.frame_off_sets[key]], model.weight[key])
        return model
    def get_weights_string(self, vals_dict):
        out = ""
        for key in self.frame_off_sets:
            out += key + ":" + str(np.round(vals_dict[key][self.frame_center + self.frame_off_sets[key]], 2))
            out += ", "
        return out
def visualize_superpose(animation:BlendshapeAnimation, superposes:list[SuperFrame], values_dict, save_video=False, video_path = "./video.mp4", pause_length = 0.2, time_to_distance_close = 0.2, speed = 1):
    loc_on_screen = np.array((0.15, 0, 0))
    def transform_mesh(mesh: o3d.geometry, delta_pos:np.array):
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) + np.expand_dims(delta_pos, axis=0))
        return mesh
    maximum_displacement_vertices, maximum_displacement_indices, all_indices = maximum_k_displacement_vertex(animation.model, k=1)
    # introduce call back functions to stop the animation
    start_t = time.time()
    t = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Face_visualize', width=1800, height=600)
    vis.get_render_option().mesh_show_wireframe = True
    # vis.rendering.Camera.Projection = 1
    animation.translate(loc_on_screen)
    vis.add_geometry(animation.eval(0))
    poses = []
    for i in range(0, len(superposes)):
        super_frame: SuperFrame = superposes[i]
        pose = copy.deepcopy(animation.model)
        pose = super_frame.set_super_frame(pose, values_dict)
        pose.translate(np.array([time_to_distance_close * (animation.ts[superposes[i].frame_center]), 0, 0]))
        pose.visualization_mesh.paint_uniform_color([0.95, 0.95, 0.95])
        # paint maximum displaced vertices with red
        vertex_colors = np.asarray(pose.visualization_mesh.vertex_colors)
        for dispalced_blendshape in super_frame.frame_off_sets:
            for v in maximum_displacement_indices[dispalced_blendshape.lower()]:
                vertex_colors[v] = np.array([1, 0, 0])
        pose.visualization_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        # colors[0] = [0, 0,]
        # print(np.asarray(pose.visualization_mesh.vertex_colors))
        vis.add_geometry(pose.eval(), reset_bounding_box=False)
        # Give each activated vertex of interest a color:
        # A[2983298329]
        poses.append(pose)
    # do the while loop and translate each object
    animation.translate(np.array([0, 0, -0.03]))
    frames = [] 
    lag_time = 0
    while t < animation.ts[-1]:
        vis.get_view_control().change_field_of_view(-90)
        if not save_video:
            dt = (time.time() - start_t - t)
        else:
            dt = 1.0/30.0
        # the lag will get reduced first before anything else happens
        lag_time -= dt
        lag_time = max(0, lag_time)
        if lag_time <= 0:
            t += dt
            # vis.get_view_control().camera_local_translate(0, dt*time_to_distance, 0)
            # animation.translate(np.array([dt*time_to_distance, 0, 0]))
            min_delta_t = 100
            for i in range(0, len(poses)):
                # print(superposes[i].frame_center)
                delta_t = np.abs(t - animation.ts[superposes[i].frame_center])
                poses[i].translate([-time_to_distance_close * dt , 0, 0])
                scaling = max(1 - delta_t, 0.07)
                poses[i].scale(scaling)
                vis.update_geometry(poses[i].eval())
                min_delta_t = min(delta_t, min_delta_t)
            vis.update_geometry(animation.eval(t))
            event = vis.poll_events()
            vis.update_renderer()
            if min_delta_t <= dt / 2:
                lag_time += pause_length
                start_t += pause_length
        # during lag time, render spheres on the face
        else:
            pass

        # capture the image if we are saving the video
        if save_video:
            im = vis.capture_screen_float_buffer(True)
            frames.append((np.asarray(im)))
    vis.destroy_window()
    if save_video:
        # if this does work, pip install upgrade "moviepy" lol
        # Define the video writer
        uint8_frames = [(f*255).astype(np.uint8) for f in frames]
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(uint8_frames, fps=30)
        clip.write_videofile(video_path)
    return frames
# visualize_superpose(animation, super_frames, values_dict, True, "./super_frames.mp4", pause_length=0.5)
def do_intersect(fin1:SharkFinCurve, fin2:SharkFinCurve, delta=2):
    # range_1 = fin1.get_range()
    # range_1 = [fin1.get_min_key_frames()[0], fin1.get_min_key_frames()[1]]
    range_1 = [fin1.get_min_key_frames()[0] - delta, fin1.get_min_key_frames()[0] + delta]
    # range_2 = fin2.get_range()
    # range_2 = [fin2.get_min_key_frames()[0], fin2.get_min_key_frames()[1]]
    range_2 = [fin2.get_min_key_frames()[0] - delta, fin2.get_min_key_frames()[0] + delta]
    
    if np.maximum(range_1[0], range_2[0]) <= np.minimum(range_1[1], range_2[1]):
        return True
    else:
        return False