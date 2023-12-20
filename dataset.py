import numpy as np
import cv2
import os
from torch.utils.data import Dataset
from cvtransforms import *
import torch
import editdistance
import json


class MyDataset(Dataset):
    letters = [
        " ",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]

    def __init__(
        self,
        video_path,
        anno_path,
        coords_path,
        file_list,
        vid_pad,
        txt_pad,
        phase,
    ):
        self.anno_path = anno_path
        self.coords_path = coords_path
        self.vid_pad = vid_pad
        self.txt_pad = txt_pad
        self.phase = phase

        with open(file_list, "r") as f:
            self.videos = [
                os.path.join(video_path, line.strip()) for line in f.readlines()
            ]

        self.data = []
        for vid in self.videos:
            items = vid.split("/")
            self.data.append((vid, items[-4], items[-1]))

    def __getitem__(self, idx):
        (vid, spk, name) = self.data[idx]
        vid = self._load_vid(vid)
        anno = self._load_anno(
            os.path.join(self.anno_path, spk, "align", name + ".align")
        )
        coord = self._load_coords(os.path.join(self.coords_path, spk, name + ".json"))

        if self.phase == "train":
            vid = HorizontalFlip(vid)

        vid = ColorNormalize(vid)

        vid_len = vid.shape[0]
        anno_len = anno.shape[0]
        vid = self._padding(vid, self.vid_pad)
        anno = self._padding(anno, self.txt_pad)
        coord = self._padding(coord, self.vid_pad)

        return {
            "vid": torch.FloatTensor(vid.transpose(3, 0, 1, 2)),
            "txt": torch.LongTensor(anno),
            "coord": torch.FloatTensor(coord),
            "txt_len": anno_len,
            "vid_len": vid_len,
        }

    def __len__(self):
        return len(self.data)

    def _load_vid(self, p):
        files = os.listdir(p)
        files = list(filter(lambda file: file.find(".jpg") != -1, files))
        files = sorted(files, key=lambda file: int(os.path.splitext(file)[0]))
        array = [cv2.imread(os.path.join(p, file)) for file in files]
        array = list(filter(lambda im: not im is None, array))
        array = [
            cv2.resize(im, (128, 64), interpolation=cv2.INTER_LANCZOS4) for im in array
        ]
        array = np.stack(array, axis=0).astype(np.float32)

        return array

    def _load_anno(self, name):
        with open(name, "r") as f:
            lines = [line.strip().split(" ") for line in f.readlines()]
            txt = [line[2] for line in lines]
            txt = list(filter(lambda s: not s.upper() in ["SIL", "SP"], txt))
        return MyDataset.txt2arr(" ".join(txt).upper(), 1)

    def _load_coords(self, name):
        # obtained from the resized image in the lip coordinate extraction
        img_width = 600
        img_height = 500
        with open(name, "r") as f:
            coords_data = json.load(f)

        coords = []
        for frame in sorted(coords_data.keys(), key=int):
            frame_coords = coords_data[frame]

            # Normalize the coordinates
            normalized_coords = []
            for x, y in zip(frame_coords[0], frame_coords[1]):
                normalized_x = x / img_width
                normalized_y = y / img_height
                normalized_coords.append((normalized_x, normalized_y))

            coords.append(normalized_coords)
        coords_array = np.array(coords, dtype=np.float32)
        return coords_array

    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)

    @staticmethod
    def txt2arr(txt, start):
        arr = []
        for c in list(txt):
            arr.append(MyDataset.letters.index(c) + start)
        return np.array(arr)

    @staticmethod
    def arr2txt(arr, start):
        txt = []
        for n in arr:
            if n >= start:
                txt.append(MyDataset.letters[n - start])
        return "".join(txt).strip()

    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if pre != n and n >= start:
                if (
                    len(txt) > 0
                    and txt[-1] == " "
                    and MyDataset.letters[n - start] == " "
                ):
                    pass
                else:
                    txt.append(MyDataset.letters[n - start])
            pre = n
        return "".join(txt).strip()

    @staticmethod
    def wer(predict, truth):
        word_pairs = [(p[0].split(" "), p[1].split(" ")) for p in zip(predict, truth)]
        wer = [1.0 * editdistance.eval(p[0], p[1]) / len(p[1]) for p in word_pairs]
        return wer

    @staticmethod
    def cer(predict, truth):
        cer = [
            1.0 * editdistance.eval(p[0], p[1]) / len(p[1]) for p in zip(predict, truth)
        ]
        return cer
