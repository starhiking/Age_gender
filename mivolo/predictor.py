from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import tqdm
from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector
from mivolo.structures import AGE_GENDER_TYPE, PersonAndFaceResult
from mivolo.data.misc import prepare_classification_images


class Predictor:
    def __init__(self, config, verbose: bool = False):
        self.detector = Detector(config.detector_weights, config.device, verbose=verbose)
        self.age_gender_model = MiVOLO(
            config.checkpoint,
            config.device,
            half=True,
            use_persons=config.with_persons,
            disable_faces=config.disable_faces,
            verbose=verbose,
        )
        self.draw = config.draw


    def plot_scrfd(self, image, scrfd_boxes, scrfd_results):
        new_image = image.copy()
        for i,(box, agr) in enumerate(zip(scrfd_boxes, scrfd_results)):
            x_min, y_min, x_max, y_max = box
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            cv2.rectangle(new_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # 根据框的大小和属性的数量，调整字体大小
            font_scale = 0.5
            font_thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            

            for j, (key, value) in enumerate(agr.items()):
                text = f"{key}: {value}"
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                cv2.rectangle(new_image, (x_min, y_min - text_height - 10 - j * (text_height + 5)), (x_min + text_width, y_min - 10 - j * (text_height + 5)), (0, 255, 0), -1)
                cv2.putText(new_image, text, (x_min, y_min - 10 - j * (text_height + 5)), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        return new_image
        

    def recognize_scrfd(self, image, scrfd_boxes):
        faces_crops = []
        im_h, im_w = image.shape[:2]
        for i, box in enumerate(scrfd_boxes):
            box[0] = min(max(0, box[0]), im_w - 1)
            box[1] = min(max(0, box[1]), im_h - 1)
            box[2] = min(max(0, box[2]), im_w - 1)
            box[3] = min(max(0, box[3]), im_h - 1)
            x1, y1, x2, y2 = box
            obj_image = image[y1:y2, x1:x2].copy()
            faces_crops.append(obj_image)
        faces_input = prepare_classification_images(
            faces_crops, self.age_gender_model.input_size, self.age_gender_model.data_config["mean"], self.age_gender_model.data_config["std"], device=self.age_gender_model.device
        )
        output = self.age_gender_model.inference(faces_input)
        scrfd_results = []
        self.age_gender_model.fill_in_results_scrfd(scrfd_results, output)
        out_im = None
        if self.draw:
            out_im = self.plot_scrfd(image, scrfd_boxes, scrfd_results)
        return scrfd_results, out_im

    def recognize(self, image: np.ndarray) -> Tuple[PersonAndFaceResult, Optional[np.ndarray]]:
        detected_objects: PersonAndFaceResult = self.detector.predict(image)
        self.age_gender_model.predict(image, detected_objects)

        out_im = None
        if self.draw:
            # plot results on image
            out_im = detected_objects.plot()

        return detected_objects, out_im

    def recognize_video(self, source: str) -> Generator:
        video_capture = cv2.VideoCapture(source)
        if not video_capture.isOpened():
            raise ValueError(f"Failed to open video source {source}")

        detected_objects_history: Dict[int, List[AGE_GENDER_TYPE]] = defaultdict(list)

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm.tqdm(range(total_frames)):
            ret, frame = video_capture.read()
            if not ret:
                break

            detected_objects: PersonAndFaceResult = self.detector.track(frame)
            self.age_gender_model.predict(frame, detected_objects)

            current_frame_objs = detected_objects.get_results_for_tracking()
            cur_persons: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[0]
            cur_faces: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[1]

            # add tr_persons and tr_faces to history
            for guid, data in cur_persons.items():
                # not useful for tracking :)
                if None not in data:
                    detected_objects_history[guid].append(data)
            for guid, data in cur_faces.items():
                if None not in data:
                    detected_objects_history[guid].append(data)

            detected_objects.set_tracked_age_gender(detected_objects_history)
            if self.draw:
                frame = detected_objects.plot()
            yield detected_objects_history, frame
