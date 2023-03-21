import os
import moviepy.editor as ed
import json
import cv2
import shutil
import numpy as np
####################################### general purpose utility functions #######################################


def get_wav_from_video(file_name, video_folder_path):
    """
    Extract the audio out of the video file, and save it as a separate wav file with the same file name
    :param file_name: Name of the file including the file extension. Does not need to include the full path
    :param video_folder_path: full path pointint to the folder containing file_name
    :return: name of the wav file
    """
    dir_files = os.listdir(video_folder_path)
    if len(dir_files) == 0:
        print("The directory is empty")
        return []
    for video in os.listdir(video_folder_path):
        # print(video)
        if video == file_name:
            video_path = os.path.join(video_folder_path, video)
            my_clip = ed.VideoFileClip(video_path)
            my_clip.audio.write_audiofile(video_path[:-3] + "wav")
    return video_path[:-3] + "wav"
def get_wav_from_video(file_name, video_folder_path=""):
    """
    Extract the audio out of the video file, and save it as a separate wav file with the same file name
    :param file_name: Name of the file including the file extension. Does not need to include the full path
    :param video_folder_path: full path pointint to the folder containing file_name
    :return: name of the wav file
    """
    if video_folder_path == "":
        video_path = file_name
        my_clip = ed.VideoFileClip(video_path)
        my_clip.audio.write_audiofile(video_path[:-3] + "wav")
        return video_path[:-3] + "wav"
    else:
        dir_files = os.listdir(video_folder_path)
        if len(dir_files) == 0:
            print("The directory is empty")
            return []
        for video in os.listdir(video_folder_path):
            # print(video)
            if video == file_name:
                video_path = os.path.join(video_folder_path, video)
                my_clip = ed.VideoFileClip(video_path)
                my_clip.audio.write_audiofile(video_path[:-3] + "wav")
        return video_path[:-3] + "wav"
def get_frames_from_video(file_name, video_folder_path, target_fps = 30, remove=False):
    # filename can just be the name of the file,
    # the video must be in the video folder_path
    frames = []
    dir_files = os.listdir(video_folder_path)
    if len(dir_files) == 0:
        print("The directory is empty")
        return []

    for video in os.listdir(video_folder_path):
        # print(video)
        if video == file_name:
            video_path = os.path.join(video_folder_path, video)
            video_folder = os.path.join(video_folder_path, video[:-4])
            try:
                # print(video_folder)
                os.mkdir(video_folder)
            except:
                if remove:
                    shutil.rmtree(video_folder, ignore_errors=True)
                    os.mkdir(video_folder)
                else:
                    dir_ls = os.listdir(video_folder)
                    counter = 0
                    for i in range(0, len(dir_ls)):
                        if dir_ls[i][-4:] == ".jpg":
                            frames.append(video_folder + "/frame%d.jpg" % counter)
                            counter = counter + 1
                    print("video to image conversion was done before, {} frames are loaded".format(len(frames)))
                    return frames
            my_clip = ed.VideoFileClip(video_path)
            my_clip.audio.write_audiofile(os.path.join(video_folder, "audio.mp3"))
            vidcap = cv2.VideoCapture(video_path)
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            meta_data = {}
            if fps <= target_fps:
                meta_data["fps"] = fps
            else:
                factor = fps/target_fps
            meta_data["fps"] = fps
            meta_data["video_path"] = video_path
            meta_data["audio_path"] = os.path.join(video_folder, "audio.mp3")
            with open(os.path.join(video_folder, "other_info.json"), 'w') as outfile:
                json.dump(meta_data, outfile)
            success, image = vidcap.read()
            count = 0
            while success:
                cv2.imwrite(video_folder + "/frame%d.jpg" % count, image)  # save frame as JPEG file
                success, image = vidcap.read()
                frames.append(video_folder + "/frame%d.jpg" % count)
                count += 1
    print("video to image conversion done")
    return frames
def split_video_to_images(file_name, video_folder_path, target_fps = 30, remove=False):
    # filename can just be the name of the file,
    # the video must be in the video folder_path
    frames = []
    dir_files = os.listdir(video_folder_path)
    if len(dir_files) == 0:
        print("The directory is empty")
        return []

    for video in os.listdir(video_folder_path):
        # print(video)
        if video == file_name:
            video_path = os.path.join(video_folder_path, video)
            video_folder = os.path.join(video_folder_path, video[:-4])
            try:
                # print(video_folder)
                os.mkdir(video_folder)
            except:
                if remove:
                    shutil.rmtree(video_folder, ignore_errors=True)
                    os.mkdir(video_folder)
                else:
                    dir_ls = os.listdir(video_folder)
                    counter = 0
                    for i in range(0, len(dir_ls)):
                        if dir_ls[i][-4:] == ".jpg":
                            frames.append(video_folder + "/frame%d.jpg" % counter)
                            counter = counter + 1
                    print("video to image conversion was done before, {} frames are loaded".format(len(frames)))
                    return frames
            my_clip = ed.VideoFileClip(video_path)
            my_clip.audio.write_audiofile(os.path.join(video_folder, "audio.mp3"))
            vidcap = cv2.VideoCapture(video_path)
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            meta_data = {}
            if fps <= target_fps:
                meta_data["fps"] = fps
            else:
                factor = fps/target_fps
            meta_data["fps"] = fps
            meta_data["video_path"] = video_path
            meta_data["audio_path"] = os.path.join(video_folder, "audio.mp3")
            with open(os.path.join(video_folder, "other_info.json"), 'w') as outfile:
                json.dump(meta_data, outfile)
            success, image = vidcap.read()
            count = 0
            while success:
                cv2.imwrite(video_folder + "/frame%d.jpg" % count, image)  # save frame as JPEG file
                success, image = vidcap.read()
                frames.append(video_folder + "/frame%d.jpg" % count)
                count += 1
    print("video to image conversion done")
    return frames
def get_point_flow(frame_t0, frame_t1, pts, pixel_sparsity=2):
    # optical flow parameter
    lk_params = dict(winSize=(9, 9),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # select a patch of pixels around the selected pts
    pixel_patch = []
    assignment = []
    for i in range(0, pts.shape[0] - 1):
        pi = pts[i]
        pj = pts[i + 1]
        bb = [[min(pi[0], pj[0]), max(pi[0], pj[0])],
              [min(pi[1], pj[1]), max(pi[1], pj[1])]]
        for x in range(int(np.floor(bb[0][0])), int(np.floor(bb[0][1])), pixel_sparsity):
            for y in range(int(np.floor(bb[1][0])), int(np.floor(bb[1][1])), pixel_sparsity):
                pt = np.array([x, y])
                if np.linalg.norm(pt - pi) < np.linalg.norm(pt - pj):
                    assignment.append(i)
                else:
                    assignment.append(i + 1)
                pixel_patch.append([x, y])
                # compute optical flow of the entire patch
    p0 = np.expand_dims(pixel_patch, axis=1).astype(np.float32)
    p1, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(frame_t0, cv2.COLOR_BGR2GRAY),
                                           cv2.cvtColor(frame_t1, cv2.COLOR_BGR2GRAY),
                                           p0, None, **lk_params)
    # determine the average velocity of pixels inside that patch
    p1 = np.array(p1)[:, 0, :]
    p0 = np.array(p0)[:, 0, :]
    delta = p1 - p0
    velocity = []
    selection = []
    pts_new = []
    for i in range(0, pts.shape[0]):
        selection.append([])
    for i in range(0, delta.shape[0]):
        selection[assignment[i]].append(delta[i])
    for i in range(0, len(selection)):
        if len(selection[i]) != 0:
            velocity.append(np.array(selection[i]).mean(axis=0))
        else:
            velocity.append(np.array(delta[i]).mean(axis=0))
        pts_new.append(pts[i] + velocity[i])
    return velocity, np.array(pts_new)
class VideoWriter():
    def __init__(self, opath, fps=25):
        # I don't think this would work with mp4 output, it probably only works with avi
        self.img_array = []
        self.opath = opath
        self.size = (-1, -1)
        self.fps = fps
    def add_frame(self, img):
        self.img_array.append(img)
        height, width, layers = img.shape
        self.size = (width, height)
    def save(self):
        out = cv2.VideoWriter(self.opath, cv2.VideoWriter_fourcc(*'DIVX'), self.fps, self.size)
        for i in range(len(self.img_array)):
            out.write(self.img_array[i])
        out.release()