import cv2
import dlib
import json
import glob
import os
from multiprocessing import Pool

LIP_COORDINATES_DIRECTORY = "lip_coordinates"
ERROR_DIRECTORY = "error_videos"

# path to the original GRID dataset whose videos are converted to frames
GRID_IMAGES_DIRECTORY = "lip/GRID_imgs"
train_unseen_list = "data/unseen_val.txt"
train_overlap_list = "data/overlap_train.txt"
test_unseen_list = "data/unseen_val.txt"
test_overlap_list = "data/overlap_val.txt"


def load_data_list(data_path, dictionary):
    with open(data_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            speaker = line.split("/")[-4]
            vid = line.split("/")[-1]
            dictionary[f"{speaker}/{vid}"] = 1
    return dictionary


def extract_lip_coordinates(detector, predictor, img_path):
    # used to preprocess the original image frames in the GRID dataset to extract the lip coordinates
    image = cv2.imread(img_path)
    image = cv2.resize(image, (600, 500))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray)
    assert len(rects) == 1
    for rect in rects:
        # extract the coordinates of the bounding box
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()

        # apply the shape predictor to the face ROI
        shape = predictor(gray, rect)
        x = []
        y = []
        for n in range(48, 68):
            x.append(shape.part(n).x)
            y.append(shape.part(n).y)
    return [x, y]


def log_error_video(video_path):
    print("Error: ", video_path)
    with open(ERROR_DIRECTORY + "/error_videos.txt", "a") as f:
        f.write(video_path + "\n")


data_dict = {}
data_dict = load_data_list(train_unseen_list, data_dict)
data_dict = load_data_list(train_overlap_list, data_dict)
data_dict = load_data_list(test_unseen_list, data_dict)
data_dict = load_data_list(test_overlap_list, data_dict)


speakers = glob.glob(GRID_IMAGES_DIRECTORY + "/*")
print(speakers[0])


def generate_lip_coordinates(speakers):
    file_path_sep = "\\"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "lip_coordinate_extraction/shape_predictor_68_face_landmarks_GTX.dat"
    )
    for speaker in speakers:
        print(speaker)
        videos = glob.glob(speaker + "/*")
        for video in videos:
            print(video)
            frames = glob.glob(video + "/*.jpg")
            if len(frames) < 50:  # filter out bad videos
                continue
            vid = {}
            try:
                frames = sorted(
                    frames,
                    key=lambda x: int(x.split(file_path_sep)[-1].split(".")[0]),
                )
                for frame in frames:
                    retry = 3
                    while retry > 0:
                        try:
                            coords = extract_lip_coordinates(detector, predictor, frame)
                            break
                        except Exception as e:
                            retry -= 1
                            print("Error: ", video)
                            print(e)
                            print("retrying...")

                    vid[frame.split(file_path_sep)[-1].split(".")[0]] = coords
                vid_path = video.split(file_path_sep)
                save_path = (
                    LIP_COORDINATES_DIRECTORY
                    + "/"
                    + vid_path[-2]
                    + "/"
                    + vid_path[-1]
                    + ".json"
                )

                if not os.path.exists(LIP_COORDINATES_DIRECTORY + "/" + vid_path[-2]):
                    os.makedirs(LIP_COORDINATES_DIRECTORY + "/" + vid_path[-2])

                with open(
                    save_path,
                    "w",
                ) as f:
                    json.dump(vid, f)
            except Exception as e:
                print(e)
                log_error_video(video)


def generate_lip_coordinates(speakers):
    file_path_sep = "\\"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "lip_coordinate_extraction/shape_predictor_68_face_landmarks_GTX.dat"
    )
    for speaker in speakers:
        print(speaker)
        videos = glob.glob(speaker + "/*")
        for video in videos:
            # if (
            #     video.split(file_path_sep)[-2] + "/" + video.split(file_path_sep)[-1]
            #     not in data_dict
            # ):
            #     continue
            print(video)
            frames = glob.glob(video + "/*.jpg")
            if len(frames) < 50:  # filter out bad videos
                continue
            vid = {}
            try:
                frames = sorted(
                    frames,
                    key=lambda x: int(x.split(file_path_sep)[-1].split(".")[0]),
                )
                for frame in frames:
                    retry = 3
                    while retry > 0:
                        try:
                            coords = extract_lip_coordinates(detector, predictor, frame)
                            break
                        except Exception as e:
                            retry -= 1
                            print("Error: ", video)
                            print(e)
                            print("retrying...")

                    vid[frame.split(file_path_sep)[-1].split(".")[0]] = coords
                vid_path = video.split(file_path_sep)
                save_path = (
                    LIP_COORDINATES_DIRECTORY
                    + "/"
                    + vid_path[-2]
                    + "/"
                    + vid_path[-1]
                    + ".json"
                )

                if not os.path.exists(LIP_COORDINATES_DIRECTORY + "/" + vid_path[-2]):
                    os.makedirs(LIP_COORDINATES_DIRECTORY + "/" + vid_path[-2])

                with open(
                    save_path,
                    "w",
                ) as f:
                    json.dump(vid, f)
            except Exception as e:
                print(e)
                log_error_video(video)


num_processes = 8

speaker_groups = []
speaker_interval = len(speakers) // num_processes
for i in range(num_processes):
    if i == 4:
        speaker_groups.append(speakers[i * speaker_interval :])
    else:
        speaker_groups.append(
            speakers[i * speaker_interval : (i + 1) * speaker_interval]
        )


if __name__ == "__main__":
    with Pool(num_processes) as p:
        p.map(generate_lip_coordinates, speaker_groups)
