from typing import List

import gdown
import numpy as np
import torch

from .data import cfg_re50
from .layers.functions import PriorBox
from .models.retinaface import RetinaFace, load_weights
from .utils.box_utils import decode, decode_landm
from .utils.nms.py_cpu_nms import py_cpu_nms


class FaceDetector(object):
    def __init__(self, weights: str = "https://drive.google.com/uc?id=14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW",
                 confidence: float = 0.02, top_k: int = 5000, nms_threshold: float = 0.4, keep_top_k: int = 750,
                 cpu: bool = False, padding: float = 0.2):
        self.confidence = confidence
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.padding = padding

        if 'http' in weights:
            weights = self.download_weights(weights)
        self.model = RetinaFace(cfg=cfg_re50, phase="test")
        self.model = load_weights(self.model, weights, load_to_cpu=cpu)
        self.model.eval()

        self.device = torch.device("cpu" if cpu else "cuda")
        self.model.to(self.device)

    @staticmethod
    def download_weights(url):
        gdown.download(url, 'weights.pth', quiet=False)
        return 'weights.pth'

    def get_faces(self, image: np.ndarray) -> (List[np.ndarray], List[float]):
        with torch.no_grad():
            img = image
            image = image.astype(np.float32)
            im_height, im_width, _ = image.shape
            scale = torch.Tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            image -= (104, 117, 123)
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).unsqueeze(0)
            image = image.to(self.device)
            scale = scale.to(self.device)


            loc, conf, landms = self.model(image)  # forward pass

            priorbox = PriorBox(cfg_re50, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg_re50['variance'])
            boxes = boxes * scale
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_re50['variance'])
            scale1 = torch.Tensor([image.shape[3], image.shape[2], image.shape[3], image.shape[2],
                                   image.shape[3], image.shape[2], image.shape[3], image.shape[2],
                                   image.shape[3], image.shape[2]])
            scale1 = scale1.to(self.device)
            landms = landms * scale1
            landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        faces = []
        scores = []
        for b in dets:
            scores.append(b[4])
            b = list(map(int, b))
            padding_height = int((b[2] - b[0]) * self.padding)
            padding_width = int((b[3] - b[1]) * self.padding)

            faces.append(img[max(int(b[1] - padding_height), 0):int(b[3]) + padding_height,
                                max(int(b[0]) - padding_width, 0):int(b[2]) + padding_width, :])
        return faces, scores
