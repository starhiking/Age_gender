# inference images and compared with the different rectangle box (scrfd and mivolo)
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
    parser.add_argument("--output", type=str, default=None, required=True, help="folder for output results")
    parser.add_argument("--detector-weights", type=str, default=None, required=True, help="Detector weights (YOLOv8).")
    parser.add_argument("--checkpoint", default="", type=str, required=True, help="path to mivolo checkpoint")

    parser.add_argument(
        "--with-persons", action="store_true", default=False, help="If set model will run with persons, if available"
    )
    parser.add_argument(
        "--disable-faces", action="store_true", default=False, help="If set model will use only persons if available"
    )

    parser.add_argument("--draw", action="store_true", default=False, help="If set, resulted images will be drawn")
    parser.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")
    
    parser.add_argument("--data_json", type=str, default=None, help="path to json file with data")
    parser.add_argument("--img_folder", type=str, default=None, help="path to json file with data")

    return parser


def main():
    parser = get_parser()
    setup_default_logging()
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.output, exist_ok=True)

    predictor = Predictor(args, verbose=True)

    input_type = InputType.Image # get_input_type(args.input)

    if input_type == InputType.Video or input_type == InputType.VideoStream:
        if not args.draw:
            raise ValueError("Video processing is only supported with --draw flag. No other way to visualize results.")

        if "youtube" in args.input:
            args.input, res, fps, yid = get_direct_video_url(args.input)
            if not args.input:
                raise ValueError(f"Failed to get direct video url {args.input}")
            outfilename = os.path.join(args.output, f"out_{yid}.avi")
        else:
            bname = os.path.splitext(os.path.basename(args.input))[0]
            outfilename = os.path.join(args.output, f"out_{bname}.avi")
            res, fps = get_local_video_info(args.input)

        if args.draw:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(outfilename, fourcc, fps, res)
            _logger.info(f"Saving result to {outfilename}..")

        for (detected_objects_history, frame) in predictor.recognize_video(args.input):
            if args.draw:
                out.write(frame)

    elif input_type == InputType.Image:
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
            detected_objects, out_im = predictor.recognize(img)
            if args.draw:
                bname = os.path.splitext(os.path.basename(img_p))[0]
                filename = os.path.join(args.output, f"out_{bname}.jpg")
                cv2.imwrite(filename, out_im)
                _logger.info(f"Saved result to {filename}")
            
            
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
            if args.draw:
                bname = os.path.splitext(os.path.basename(img_p))[0]
                filename = os.path.join(args.output, f"out_{bname}_scrfd.jpg")
                cv2.imwrite(filename, out_im)
                #_logger.info(f"Saved result to {filename}")

if __name__ == "__main__":
    main()
