import numpy as np
import torch

from .deep.feature_extractor import Extractor, FastReIDExtractor

from .sort.preprocessing import non_max_suppression
# from .sort.detection import Detection
from .match_metrics import NearestNeighborDistanceMetric
from .tracker import Tracker
from modules.detectors import FasterRCNN
from modules.backbones import resnet50_fpn_backbone


__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self,
                 detector_config,
                 model_path, 
                 model_config=None, 
                 max_cosine_distance=0.2, 
                 min_confidence=0.3, 
                 nms_max_overlap=1.0,
                 max_iou_distance=0.7, 
                 max_age=70, 
                 n_init=3, 
                 nn_budget=100, 
                 use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap


        # gy：构造detector
        self.detector = FasterRCNN(backbone=resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d), num_classes=num_classes, rpn_score_thresh=0.5)

        # gy：model_config为空使用的是xxxx，否则使用fastreid
        if model_config is None:
            self.extractor = Extractor(model_path, use_cuda=use_cuda)
        else:
            self.extractor = FastReIDExtractor(model_config, model_path, use_cuda=use_cuda)



        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, ori_img):

        # 检测
        predictions = self.detector(ori_img)
        bbox_xywh = predictions["boxes"]
        cls_ids = predictions["labels"]
        cls_conf = predictions["scores"]
        # select person class， gy：筛选出类别为0的box
        label_mask = cls_ids == 0
        bbox_xywh = bbox_xywh[label_mask]
        # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
        bbox_xywh[:, 3:] *= 1.2
        cls_conf = cls_conf[label_mask]
        self.height, self.width = ori_img.shape[:2]
        # generate detections

        # gy：使用reid提特征
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh) #gy：坐标转换
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences) if
                      conf > self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)# 非极大值抑制
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        # 如果box非空
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
