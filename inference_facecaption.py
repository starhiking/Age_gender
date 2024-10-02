## inference images with scrfd boxes, and output json file
import argparse
import json
import logging
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm
import yt_dlp
from mivolo.data.data_reader import InputType, get_all_files, get_input_type
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging

_logger = logging.getLogger("inference")


def get_direct_video_url(video_url):
    ydl_opts = {
        "format": "bestvideo",
        "quiet": True,  # Suppress terminal output (remove this line if you want to see the log)
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)

        if "url" in info_dict:
            direct_url = info_dict["url"]
            resolution = (info_dict["width"], info_dict["height"])
            fps = info_dict["fps"]
            yid = info_dict["id"]
            return direct_url, resolution, fps, yid

    return None, None, None, None


def get_local_video_info(vid_uri):
    cap = cv2.VideoCapture(vid_uri)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source {vid_uri}")
    res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return res, fps


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Inference")
    #parser.add_argument("--input", type=str, default=None, required=True, help="image file or folder with images")
    parser.add_argument("--output", type=str, default=None, help="folder for output results")
    parser.add_argument("--detector-weights", type=str, default="models/yolov8x_person_face.pt", help="Detector weights (YOLOv8).")
    parser.add_argument("--checkpoint", default="models/model_imdb_age_gender_4.22.pth.tar", type=str, help="path to mivolo checkpoint")

    parser.add_argument(
        "--with-persons", action="store_true", default=False, help="If set model will run with persons, if available"
    )
    parser.add_argument(
        "--disable-faces", action="store_true", default=False, help="If set model will use only persons if available"
    )

    parser.add_argument("--draw", action="store_true", default=False, help="If set, resulted images will be drawn")
    parser.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")
    
    parser.add_argument("--data_json", type=str, default=None, help="path to json file with data")
    parser.add_argument("--img_folder", type=str, default="/mnt/data/lanxing/faceCaption5M/images/images", help="path to json file with data")
    parser.add_argument("--result_json", type=str, default=None, help="path to json file with data")

    return parser


def main():
    parser = get_parser()
    setup_default_logging()
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    predictor = Predictor(args, verbose=True)

    #image_files = get_all_files(args.input) if os.path.isdir(args.input) else [args.input]
    
    with open(args.data_json, 'r') as f:
        data = json.load(f)
    

    
    for data_i in tqdm(data):
        img_name = data_i["path"]
        img_p = os.path.join(args.img_folder, img_name)
        if not os.path.exists(img_p):
            print(f"Image {img_p} does not exist")
            exit()

        img = cv2.imread(img_p)
        faces = data_i["result"]
        scrfd_boxes = []
        for i, face_result in enumerate(faces):
            x_min, y_min, w, h = face_result['facial_area']
            x_min, y_min, w, h = float(x_min), float(y_min), float(w), float(h)
            x_min, y_min, w, h = int(x_min), int(y_min), int(w), int(h)
            x_max = x_min + w
            y_max = y_min + h
            scrfd_boxes.append([x_min, y_min, x_max, y_max])
        age_genders, out_im = predictor.recognize_scrfd(img, scrfd_boxes)
        
        for face_result, age_gender in zip(faces, age_genders):
            face_result["age"] = age_gender["age"]
            face_result["gender"] = age_gender["gender"]
            
        if args.draw:
            os.makedirs(args.output, exist_ok=True)
            bname = os.path.splitext(os.path.basename(img_p))[0]
            filename = os.path.join(args.output, f"out_{bname}_scrfd.jpg")
            cv2.imwrite(filename, out_im)
            _logger.info(f"Saved result to {filename}")
    
    
    os.makedirs(os.path.dirname(args.result_json), exist_ok=True)
    with open(args.result_json, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()
