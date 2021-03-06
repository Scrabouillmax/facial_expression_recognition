import time
import cv2
import os
import json
import pandas as pd
import numpy as np

from dataset_tools import string_to_pilimage

with open('config.json') as json_file:
    config = json.load(json_file)


def get_img_path():
    """
    :return: list of all paths for images contained in the folder config["path_images"]:
    """
    paths = []
    for r, d, f in os.walk(config["path_images"]):
        for file in f:
            if any(extension in file for extension in ['.jpeg', '.jpg', '.png']):
                paths.append(os.path.join(r, file))
    return paths


def load_cv_imgs(paths):
    """
    :return: load opencv images from list of image paths.
    """
    imgs = []
    for index, path in enumerate(paths):
        image = cv2.imread(path)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgs.append(gray_img)

    return imgs


def load_cv_images_from_fer():
    """
    :return: list of opencv images from the fer csv file.
    """
    cv_imgs = []
    all_data = pd.read_csv(config["path"], header=0, nrows=30000)
    for i in range(30000):
        pixels = all_data.loc[i]['pixels']
        pil_img = string_to_pilimage(pixels)
        cv_imgs.append(np.array(pil_img))

    print("loaded images from csv")

    return cv_imgs


def crop_faces(cv_imgs):
    """
    Return an array of coordinates of faces, one face per image.
    If none or several faces were found on an image, the coordinates for this image are None.

        Tested parameters for the face recognition with opencv on the FER dataset :
        Params:
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20)
        Results:
            No face:  14257
            One face:  15733
            Several faces:  10

        Params:
            scaleFactor=1.01,
            minNeighbors=3,
            minSize=(20, 20)
        Results:
            No face:  9601
            One face:  20242
            Several faces:  157

    :param cv_imgs: opencv image array.
    :return: array of coordinates of the face for each input image (which can contain be None values).
    """
    faces_coords = []
    img_with_several_faces = 0
    img_with_no_face = 0
    img_with_one_face = 0
    for index, image in enumerate(cv_imgs):
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            image,
            scaleFactor=1.01,
            minNeighbors=3,
            minSize=(20, 20)
        )

        if len(faces) == 1:
            img_with_one_face += 1
            (x, y, w, h) = faces[0]
            faces_coords.append((x, y, w, h))
        else:
            faces_coords.append(None)
            if len(faces) == 0:
                img_with_no_face += 1
            else:
                img_with_several_faces += 1

    return faces_coords


def crop_cv_img(img, x, y, w, h):
    return img[y:y + h, x:x + w]


def crop_csv_dataset(input_csv_path, output_csv_path):
    """
    Adds a column named "face" to the csv dataset with the cropped face for all the dataset, when the face is found.
    Results on fer:
        {
            ‘imgs_removed’: 11721,
            ‘imgs_kept’: 24167,
            ‘imgs_dropped_per_class’: {
                ‘Angry’: 1668,
                ‘Disgust’: 185,
                ‘Fear’: 1915,
                ‘Happy’: 2535,
                ‘Sad’: 2767,
                ‘Surprise’: 1075,
                ‘Neutral’: 1576
            },
            ‘imgs_kept_per_class’: {
                ‘Angry’: 3286,
                ‘Disgust’: 362,
                ‘Fear’: 3206,
                ‘Happy’: 6454,
                ‘Sad’: 3310,
                ‘Surprise’: 2927,
                ‘Neutral’: 4622
            }
        }
    Results on ferplus:
        {
            'imgs_removed': 11387,
            'imgs_kept': 23886,
            'imgs_dropped_per_class': {
                'Angry': 1260,
                'Disgust': 81,
                'Fear': 320,
                'Happy': 2594,
                'Sad': 2301,
                'Surprise': 1208,
                'Neutral': 3623
            },
            'imgs_kept_per_class': {
                'Angry': 2149,
                'Disgust': 223,
                'Fear': 696,
                'Happy': 6853,
                'Sad': 2550,
                'Surprise': 3142,
                'Neutral': 8273
            }
        }
    :param input_csv_path: csv dataset path
    :param output_csv_path: where to write the output csv
    :return:
    """
    initial_time = time.time()
    all_data = pd.read_csv(input_csv_path, header=0)
    stats = {
        "progress": 0,
        "imgs_removed": 0,
        "imgs_kept": 0,
        "imgs_dropped_per_class": {
            label: 0 for label in config["catslist"]
        },
        "imgs_kept_per_class": {
            label: 0 for label in config["catslist"]
        }
    }

    def crop_face_for_img(row, *args):
        (stats,) = args
        stats["progress"] += 1
        if stats["progress"] % 100 == 0:
            print("Duration so far", time.time() - initial_time, "Progress: ", int(stats["progress"]))
            print("Stats so far", stats)

        pixelstring = row["pixels"]
        img = np.array(string_to_pilimage(pixelstring))
        [faces_coords] = crop_faces([img])

        if faces_coords is None:
            stats["imgs_dropped_per_class"][config["catslist"][row["emotion"]]] += 1
            stats["imgs_removed"] += 1
            return None

        stats["imgs_kept_per_class"][config["catslist"][row["emotion"]]] += 1
        stats["imgs_kept"] += 1
        (x, y, w, h) = faces_coords

        cropped_cv_img = crop_cv_img(img, x, y, w, h)
        resized_cv_img = cv2.resize(cropped_cv_img, (48, 48))

        pixels = resized_cv_img.flatten().tolist()
        row["face"] = " ".join(map(str, pixels))
        return row

    all_data = all_data.apply(crop_face_for_img, args=(stats,), axis=1)
    all_data.dropna(inplace=True)
    all_data["emotion"] = all_data["emotion"].astype(int)
    for label in config["catslist"]:
        if label in all_data:
            all_data[label] = all_data[label].astype(int)
    all_data.to_csv(output_csv_path, index=False)
    print(stats)
