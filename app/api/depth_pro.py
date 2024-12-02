import gc
from copy import deepcopy
from itertools import groupby
from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import torch
from depth_pro import depth_pro
from depth_pro.depth_pro import DepthProConfig, DEFAULT_MONODEPTH_CONFIG_DICT

from api.patches import DEVICE
from models import PredictResponse, BoxObjects, DepthResponseWithBox, DepthResponseImage
from utils import load_video_from_path, is_base64_string, base64_to_image_with_size, load_image_from_path, tqdm_log


class DepthPro:
    def __init__(self,
                 checkpoints_path: Union[str, Path]):
        self.checkpoints_path = checkpoints_path
        self.device = DEVICE
        self.model = None
        self.transform = None

    def __init_model(self):
        if self.model is None:
            self.config: DepthProConfig = deepcopy(DEFAULT_MONODEPTH_CONFIG_DICT)
            self.config.checkpoint_uri = self.checkpoints_path
            model, transform = depth_pro.create_model_and_transforms(
                config=self.config,
                device=self.device
            )
            self.model = model
            self.transform = transform

    def unload_model_after_stream(self):
        del self.model
        del self.transform
        self.model = None
        self.transform = None
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def call_model(self,
                   images: Optional[List[str]] = None,
                   video: Optional[str] = None,
                   scale_factor=None,
                   start_second=None,
                   end_second=None,
                   focal_length_px=None,
                   box_objects: Optional[List[BoxObjects]] = None):
        images_pillow_with_size = self.__read_images(images=images, video=video,
                                                     scale_factor=scale_factor,
                                                     start_second=start_second,
                                                     end_second=end_second)
        if box_objects:
            boxes_by_frame = self.__group_box_objects_by_frame(box_objects=box_objects)
            has_boxes = True
        else:
            boxes_by_frame = {}
            has_boxes = False

        images_boxes = ((frame_idx, image_pil, boxes_by_frame.get(frame_idx)) for
                        frame_idx, (image_pil, image_size) in
                        enumerate(images_pillow_with_size))
        depth_f_px_gen = ((frame_idx, self.__call_model(image=image_pil,
                                                        focal_length_px=focal_length_px,
                                                        boxes=boxes, has_boxes=has_boxes)) for
                          frame_idx, image_pil, boxes in
                          tqdm_log(images_boxes, log_level='INFO',
                                   desc='Perform Inference with Stream'))
        response = (PredictResponse(response={frame_idx: resp}) for frame_idx, resp in depth_f_px_gen)
        return response

    def __call_model(self, image, focal_length_px, boxes, has_boxes):
        self.__init_model()
        with torch.inference_mode():
            self.model.eval()
            image_arr = np.array(image)
            image_arr = self.transform(image_arr)
            prediction = self.model.infer(image_arr, f_px=focal_length_px)
            depth = prediction["depth"]  # Depth in [m].
            if not focal_length_px:
                focal_length_px = prediction["focallength_px"]  # Focal length in pixels.
            depth_numpy: np.ndarray = depth.cpu().numpy()
            if has_boxes and boxes:
                return self.__get_mean_distance_in_bbox(focal_length_px, depth_numpy, boxes)
            elif has_boxes:
                return []
            else:
                return [DepthResponseImage(dept_matrix=depth_numpy.tolist(), focal_length_px=focal_length_px)]

    @staticmethod
    def __group_box_objects_by_frame(box_objects: List[BoxObjects]):
        items_sorted = sorted(box_objects, key=lambda x: x.frame)
        return {frame: list(boxes) for frame, boxes in groupby(items_sorted, key=lambda x: x.frame)}

    @staticmethod
    def __get_mean_distance_in_bbox(focal_length_px, depth_array, boxes):
        depths = []
        for box_idx in range(len(boxes)):
            x_min, y_min, x_max, y_max = [int(value) for value in boxes[box_idx].bbox]

            # Slice the array to get the values within the bounding box
            values_inside_box = (depth_array[y_min:y_max, x_min:x_max]).mean()

            depths.append(DepthResponseWithBox(object_id=boxes[box_idx].object_id, mean_distance=values_inside_box,
                                               focal_length_px=focal_length_px))
        return depths

    @staticmethod
    def __read_images(images: Optional[List[str]] = None,
                      video: Optional[str] = None,
                      scale_factor=None,
                      start_second=None,
                      end_second=None):
        if video is not None:
            images_pillow_with_size = load_video_from_path(video, scale_factor, start_second, end_second)
        elif images is not None and is_base64_string(images[0]):
            images_pillow_with_size = [base64_to_image_with_size(image, scale_factor) for image in images]
        else:
            images_pillow_with_size = [load_image_from_path(image_path, scale_factor) for image_path in images]
        return images_pillow_with_size
