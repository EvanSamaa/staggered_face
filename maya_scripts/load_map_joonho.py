import json
def load_apple_motion(motion_data_path = ""):
    neck_map = {"HeadYaw":"Y", "HeadPitch":"X", "HeadRoll":"Z"}
    name_to_mesh = {}
    mesh_to_name = {}
    # load model
    with open("C:/Users/evansamaa/Desktop/staggered_face/models/joonho_mesh_number_to_AU_name.json") as f:
        mesh_to_name = json.load(f)
    with open("C:/Users/evansamaa/Desktop/staggered_face/models/joonho_AU_name_to_mesh_number.json") as f:
        name_to_mesh = json.load(f)
    # load data from file
    labels = []
    raw_data = []
    head_rotation_data = []
    head_rotation_labels = []
    data = []
    times = []
    with open(motion_data_path) as f:
        labels = f.readline()
        labels = labels.split(",")
        head_rotation_labels = labels[-9:-6]
        labels = labels[2:-10]
        raw_data = f.readlines()
    for i in range(0, len(raw_data)):
        frame_time = raw_data[i].split(",")[0]
        frame_time_list = frame_time.split(":")
        frame_hour, frame_minute, frame_second, frame_milisecond = frame_time_list
        frame_hour = float(frame_hour)
        frame_minute = float(frame_minute)
        frame_second = float(frame_second)
        frame_milisecond = float(frame_milisecond)
        frame_time = frame_milisecond/60.0 + frame_second + frame_minute * 60 + frame_hour * 3600
        
        frame_data = raw_data[i].split(",")
        if len(frame_data) > 20:
            times.append(frame_time)
            data.append(frame_data[2: -10])
        head_rotation_data.append(frame_data[-9: -6])
        for j in range(len(head_rotation_data[-1])):
            head_rotation_data[-1][j] = float(head_rotation_data[-1][j])
            
    start_time = times[0]
    for i in range(0, len(times)):
        times[i] = times[i] - start_time
    # compute curve using loaded data:
    for i in range(0, len(times)):
        for j in range(0, len(labels)):
            name = labels[j]
            weight_name = name_to_mesh[str(name).lower()]
            cmds.setKeyframe("blendShape1.{}".format(weight_name), v=float(data[i][j]),
                                 t=times[i] * mel.eval('float $fps = `currentTimeUnitToFPS`'))
        for j in range(0, len(head_rotation_labels)):
            name = neck_map[head_rotation_labels[j]]
            # print(head_rotation_data)
            cmds.setKeyframe("Neutral:Mesh.rotate{}".format(name), v=float(head_rotation_data[i][j]),
                                 t=times[i] * mel.eval('float $fps = `currentTimeUnitToFPS`'))

load_apple_motion("C:/Users/evansamaa/Desktop/staggered_face/data/live_link_data/20221022_MySlate_8/MySlate_8_Evans_Iphone.csv")