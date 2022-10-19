def load_apple_motion(motion_data_path = ""):
    name_to_mesh = {}
    mesh_to_name = {}
    # load model
    with open("C:/Users/evan1/Documents/staggered_face/models/joonho_mesh_number_to_AU_name.json") as f:
        mesh_to_name = json.load(f)
    with open("C:/Users/evan1/Documents/staggered_face/models/joonho_AU_name_to_mesh_number.json") as f:
        name_to_mesh = json.load(f)
    # load data from file
    columns = []
    raw_data = []
    data = []
    times = []
    with open(motion_data_path) as f:
        labels = f.readline()
        columns = labels.split(",")
        columns = columns[2:-10]
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
    start_time = times[0]
    for i in range(0, len(times)):
        times[i] = times[i] - start_time
    # compute curve using loaded data:
    for i in range(0, len(times)):
        for j in range(0, len(columns)):
            name = columns[j]
            weight_name = name_to_mesh[str(name).lower()]
            cmds.setKeyframe("blendShape1.{}".format(weight_name), v=float(data[i][j]),
                                 t=times[i] * mel.eval('float $fps = `currentTimeUnitToFPS`'))

load_apple_motion("D:/MASC/kappa2p0/data/live_link_data/LiveLinkFace_20220926_MySlate_7_Evans_Iphone/MySlate_7_Evans_Iphone.csv")