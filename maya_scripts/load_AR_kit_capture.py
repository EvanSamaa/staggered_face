import json


calibration_data_path = "/Volumes/EVAN_DISK/MASC/staggered_face/Pilot_Experimental_study/Iphone_AR_kit/evan_take_june18_short/MySlate_14_iPhone_cal.csv"


def load_apple_motion(motion_data_path = "", recordingfps=30, calibration_data_path=""):
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
            cal_data.append(frame_cal_data[2:])
    start_time = times[0]
    for i in range(0, len(times)):
        times[i] = times[i] - start_time
    # compute curve using loaded data: 
    for i in range(0, len(times)):
        for j in range(0, len(columns)-10):
            name = columns[j].lower()
            try:
                weight_name = name_to_mesh[name]
            except:
                if name[-2:] == "_l":
                    name = name[:-2] + "left"
                else:
                    name = name[:-2] + "right"
                weight_name = name_to_mesh[name]
            cmds.setKeyframe("blendShape1.{}".format(weight_name), v=float(cal_data[i][j]),
                                 t=times[i] * mel.eval('float $fps = `currentTimeUnitToFPS`'))
        cmds.setKeyframe("Neutral:Mesh.{}".format("rotateY"), v=(float(cal_data[i][-9]))*-90,
                     t=times[i] * mel.eval('float $fps = `currentTimeUnitToFPS`'))
        cmds.setKeyframe("Neutral:Mesh.{}".format("rotateX"), v=(float(cal_data[i][-8]))*-70,
                     t=times[i] * mel.eval('float $fps = `currentTimeUnitToFPS`'))
        cmds.setKeyframe("Neutral:Mesh.{}".format("rotateZ"), v=(float(cal_data[i][-7]))*-45,
                     t=times[i] * mel.eval('float $fps = `currentTimeUnitToFPS`'))
    

load_apple_motion(calibration_data_path, 30, calibration_data_path)