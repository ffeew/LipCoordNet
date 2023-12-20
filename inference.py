import argparse
import os
from model import LipCoordNet
from dataset import MyDataset
import torch
import cv2
import face_alignment
import numpy as np
import dlib
import glob


def get_position(size, padding=0.25):
    x = [
        0.000213256,
        0.0752622,
        0.18113,
        0.29077,
        0.393397,
        0.586856,
        0.689483,
        0.799124,
        0.904991,
        0.98004,
        0.490127,
        0.490127,
        0.490127,
        0.490127,
        0.36688,
        0.426036,
        0.490127,
        0.554217,
        0.613373,
        0.121737,
        0.187122,
        0.265825,
        0.334606,
        0.260918,
        0.182743,
        0.645647,
        0.714428,
        0.793132,
        0.858516,
        0.79751,
        0.719335,
        0.254149,
        0.340985,
        0.428858,
        0.490127,
        0.551395,
        0.639268,
        0.726104,
        0.642159,
        0.556721,
        0.490127,
        0.423532,
        0.338094,
        0.290379,
        0.428096,
        0.490127,
        0.552157,
        0.689874,
        0.553364,
        0.490127,
        0.42689,
    ]

    y = [
        0.106454,
        0.038915,
        0.0187482,
        0.0344891,
        0.0773906,
        0.0773906,
        0.0344891,
        0.0187482,
        0.038915,
        0.106454,
        0.203352,
        0.307009,
        0.409805,
        0.515625,
        0.587326,
        0.609345,
        0.628106,
        0.609345,
        0.587326,
        0.216423,
        0.178758,
        0.179852,
        0.231733,
        0.245099,
        0.244077,
        0.231733,
        0.179852,
        0.178758,
        0.216423,
        0.244077,
        0.245099,
        0.780233,
        0.745405,
        0.727388,
        0.742578,
        0.727388,
        0.745405,
        0.780233,
        0.864805,
        0.902192,
        0.909281,
        0.902192,
        0.864805,
        0.784792,
        0.778746,
        0.785343,
        0.778746,
        0.784792,
        0.824182,
        0.831803,
        0.824182,
    ]

    x, y = np.array(x), np.array(y)

    x = (x + padding) / (2 * padding + 1)
    y = (y + padding) / (2 * padding + 1)
    x = x * size
    y = y * size
    return np.array(list(zip(x, y)))


def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack(
        [
            np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)),
            np.matrix([0.0, 0.0, 1.0]),
        ]
    )


def load_video(file, device: str):
    # create the samples directory if it doesn't exist
    if not os.path.exists("samples"):
        os.makedirs("samples")

    p = os.path.join("samples")
    output = os.path.join("samples", "%04d.jpg")
    cmd = "ffmpeg -hide_banner -loglevel error -i {} -qscale:v 2 -r 25 {}".format(
        file, output
    )
    os.system(cmd)

    files = os.listdir(p)
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

    array = [cv2.imread(os.path.join(p, file)) for file in files]

    array = list(filter(lambda im: not im is None, array))

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=False, device=device
    )
    points = [fa.get_landmarks(I) for I in array]

    front256 = get_position(256)
    video = []
    for point, scene in zip(points, array):
        if point is not None:
            shape = np.array(point[0])
            shape = shape[17:]
            M = transformation_from_points(np.matrix(shape), np.matrix(front256))

            img = cv2.warpAffine(scene, M[:2], (256, 256))
            (x, y) = front256[-20:].mean(0).astype(np.int32)
            w = 160 // 2
            img = img[y - w // 2 : y + w // 2, x - w : x + w, ...]
            img = cv2.resize(img, (128, 64))
            video.append(img)

    video = np.stack(video, axis=0).astype(np.float32)
    video = torch.FloatTensor(video.transpose(3, 0, 1, 2)) / 255.0

    return video


def extract_lip_coordinates(detector, predictor, img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (600, 500))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray)
    retries = 3
    while retries > 0:
        try:
            assert len(rects) == 1
            break
        except AssertionError as e:
            retries -= 1

    for rect in rects:
        # apply the shape predictor to the face ROI
        shape = predictor(gray, rect)
        x = []
        y = []
        for n in range(48, 68):
            x.append(shape.part(n).x)
            y.append(shape.part(n).y)
    return [x, y]


def generate_lip_coordinates(frame_images_directory, detector, predictor):
    frames = glob.glob(frame_images_directory + "/*.jpg")
    frames.sort()

    img = cv2.imread(frames[0])
    height, width, layers = img.shape

    coords = []
    for frame in frames:
        x_coords, y_coords = extract_lip_coordinates(detector, predictor, frame)
        normalized_coords = []
        for x, y in zip(x_coords, y_coords):
            normalized_x = x / width
            normalized_y = y / height
            normalized_coords.append((normalized_x, normalized_y))
        coords.append(normalized_coords)
    coords_array = np.array(coords, dtype=np.float32)
    coords_array = torch.from_numpy(coords_array)
    return coords_array


def ctc_decode(y):
    y = y.argmax(-1)
    t = y.size(0)
    result = []
    for i in range(t + 1):
        result.append(MyDataset.ctc_arr2txt(y[:i], start=1))
    return result


def output_video(p, txt, output_path):
    files = os.listdir(p)
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

    font = cv2.FONT_HERSHEY_SIMPLEX

    for file, line in zip(files, txt):
        img = cv2.imread(os.path.join(p, file))
        h, w, _ = img.shape
        img = cv2.putText(
            img, line, (w // 8, 11 * h // 12), font, 1.2, (0, 0, 0), 3, cv2.LINE_AA
        )
        img = cv2.putText(
            img,
            line,
            (w // 8, 11 * h // 12),
            font,
            1.2,
            (255, 255, 255),
            0,
            cv2.LINE_AA,
        )
        h = h // 2
        w = w // 2
        img = cv2.resize(img, (w, h))
        cv2.imwrite(os.path.join(p, file), img)

    # create the output_videos directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output = os.path.join(output_path, "output.mp4")
    cmd = "ffmpeg -hide_banner -loglevel error -y -i {}/%04d.jpg -r 25 {}".format(
        p, output
    )
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="pretrain/LipCoordNet_coords_loss_0.025581153109669685_wer_0.01746208431890914_cer_0.006488426950253695.pt",
        help="path to the weights file",
    )
    parser.add_argument(
        "--input_video",
        type=str,
        help="path to the input video frames",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to run the model on",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="output_videos",
        help="directory to save the output video",
    )

    args = parser.parse_args()

    # validate if device is valid
    if args.device not in ("cuda", "cpu"):
        raise ValueError("Invalid device, must be either cuda or cpu")

    device = args.device

    # load model
    model = LipCoordNet()
    model.load_state_dict(torch.load(args.weights))
    model = model.to(device)
    model.eval()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "lip_coordinate_extraction/shape_predictor_68_face_landmarks_GTX.dat"
    )

    # load video
    video = load_video(args.input_video, device)

    # generate lip coordinates
    coords = generate_lip_coordinates("samples", detector, predictor)

    pred = model(video[None, ...].to(device), coords[None, ...].to(device))
    output = ctc_decode(pred[0])
    print(output[-1])
    output_video("samples", output, args.output_path)


if __name__ == "__main__":
    main()
