import os
import moviepy.editor as ed
import json


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
