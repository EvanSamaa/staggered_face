import json
import time
from matplotlib import pyplot as plt
if __name__ == "__main__":
    # load_data
    data_path = "C:/Users/evansamaa/Desktop/Staggered_faces/data/live_link_data/LiveLinkFace_20220926_MySlate_7_Evans_Iphone/MySlate_7_Evans_Iphone.csv"
    name_to_mesh = {}
    mesh_to_name = {}
    # load model
    with open("C:/Users/evansamaa/Desktop/Staggered_faces/models/mesh_number_to_AU_name.json") as f:
        mesh_to_name = json.load(f)
    with open("C:/Users/evansamaa/Desktop/Staggered_faces/models/AU_name_to_mesh_number.json") as f:
        name_to_mesh = json.load(f)
    # load data from file
    columns = []
    data = []
    times = []
    with open(data_path) as f:
        labels = f.readline()
        columns = labels.split(",")
        columns = columns[2:-10]
        data = f.readlines()
    for i in range(0, len(data)):
        print(data[i].split(",")[0])
        frame_time = data[i].split(",")[0]
        frame_time_list = frame_time.split(":")
        frame_hour, frame_minute, frame_second, frame_milisecond = frame_time_list
        frame_hour = float(frame_hour)
        frame_minute = float(frame_minute)
        frame_second = float(frame_second)
        frame_milisecond = float(frame_milisecond)
        frame_time = frame_milisecond / 60 + frame_second + frame_minute * 60 + frame_hour * 3600
        times.append(frame_time)
        data[i] = data[i].split(",")[2: -10]
    start_time = times[0]
    for i in range(0, len(times)):
        times[i] = times[i] - start_time
    # compute curve using loaded data:
    length = []
    for i in range(0, len(data)):
        length.append(len(data[i]))
    plt.plot(length)
    plt.show()
