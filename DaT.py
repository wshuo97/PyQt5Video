import argparse
import torch
import numpy as np
import cv2
from deep_sort import build_tracker3
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords


class DaT:
    def __init__(self, detWeights="./weights/yolov5s.pt", pTrackPath="./weights/ckpt.net64do.t7",
                 vTrackPath="./weights/ckpt.net128do.t7", pnum=64, vnum=164):
        self.opt = self.getOpt()
        opt = self.opt
        self.weights = detWeights
        self.pweight, self.vweight = pTrackPath, vTrackPath
        self.use_cuda = torch.cuda.is_available()
        self.device = opt.device if self.use_cuda else "cpu"
        self.pnum = pnum
        self.vnum = vnum
        self.img_size = opt.img_size

    def setDetector(self, detWeights):
        self.weights = detWeights

    def newDetector(self, detWeights="./weights/yolov5s.pt"):
        self.weights = detWeights
        self.model = attempt_load(
            self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        # imgsz = check_img_size(imgsz, s=stride)  # check img_size

    def newTracker(self):
        opt = self.opt
        self.ptracker = build_tracker3(
            self.pweight, self.pnum, opt, self.use_cuda, self.device, (64, 128))
        self.vtracker = build_tracker3(
            self.vweight, self.vnum, opt, self.use_cuda, self.device, (128, 128))

    def detect(self, im0s):
        # img = im0s.copy()
        img = self.letterbox(im0s, self.img_size,
                             stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        opt = self.opt
        img = torch.Tensor(img).to(self.device)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)

        bbox_xywh = []
        cls_ids = None
        cls_conf = None
        # print(pred)
        for _, det in enumerate(pred):
            im0 = im0s.copy()
            if len(det):
                ret = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                ret[:, 2] = ret[:, 2] - ret[:, 0]
                ret[:, 3] = ret[:, 3] - ret[:, 1]
                ret[:, 0] = ret[:, 0] + ret[:, 2]/2
                ret[:, 1] = ret[:, 1] + ret[:, 3]/2

                clsl = det[:, 5]
                clsc = det[:, 4]

                bbox_xywh = np.asarray(ret.cpu())
                cls_ids = np.asarray(clsl.cpu())
                cls_conf = np.asarray(clsc.cpu())
        mask = (cls_ids == 0) + (cls_ids == 1) + (cls_ids == 2) + (cls_ids == 3) + \
            (cls_ids == 4) + (cls_ids == 5) + \
            (cls_ids == 6) + (cls_ids == 7)
        vmask = (cls_ids == 1) + (cls_ids == 2) + (cls_ids == 3) + \
            (cls_ids == 4) + (cls_ids == 5) + \
            (cls_ids == 6) + (cls_ids == 7)
        bbox_xywh = bbox_xywh[mask]
        cls_conf = cls_conf[mask]
        cls_ids[vmask] = 1
        cls_ids = cls_ids[mask]
        bbox_xyxy = bbox_xywh.copy()
        bbox_xyxy[:, 0] = bbox_xywh[:, 0]-bbox_xywh[:, 2]/2
        bbox_xyxy[:, 1] = bbox_xywh[:, 1]-bbox_xywh[:, 3]/2
        bbox_xyxy[:, 2] = bbox_xywh[:, 0]+bbox_xywh[:, 2]/2
        bbox_xyxy[:, 3] = bbox_xywh[:, 1]+bbox_xywh[:, 3]/2
        detRes = []
        for i in range(len(cls_ids)):
            detRes.append((bbox_xyxy[i][0], bbox_xyxy[i][1],
                          bbox_xyxy[i][2], bbox_xyxy[i][3], cls_ids[i]))
        # detRes = [detRes]
        # print(detRes)
        return (bbox_xywh, cls_ids, cls_conf, [detRes])

    def track(self, bbox_xywh, cls_ids, cls_conf, im0s):
        poutputs = []
        voutputs = []

        if len(bbox_xywh) > 0:
            pmask = (cls_ids == 0)
            pbbox_xywh = bbox_xywh[pmask]
            pcls_conf = cls_conf[pmask]
            pcls_ids = cls_ids[pmask]
            poutputs = self.ptracker.update(bbox_xywh=pbbox_xywh,
                                            classid=pcls_ids,
                                            confidences=pcls_conf,
                                            ori_img=im0s)
            vmask = (cls_ids != 0)
            vbbox_xywh = bbox_xywh[vmask]
            vcls_conf = cls_conf[vmask]
            vcls_ids = cls_ids[vmask]
            voutputs = self.vtracker.update(bbox_xywh=vbbox_xywh,
                                            classid=vcls_ids,
                                            confidences=vcls_conf,
                                            ori_img=im0s)

        return [poutputs, voutputs]

    def getOpt(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--img-size', type=int, default=640)
        parser.add_argument('--device', type=str, default="cuda:1")
        parser.add_argument('--save-txt', type=bool, default=True)
        parser.add_argument('--save-conf', action='store_true')
        parser.add_argument('--project', default='runs/detect')
        parser.add_argument('--name', default='exp')
        parser.add_argument('--exist-ok', action='store_true')
        parser.add_argument('--line-thickness', default=3)
        parser.add_argument('--pnum', type=int, default=490)
        parser.add_argument('--vnum', type=int, default=6)

        parser.add_argument('--augment', action='store_true')
        parser.add_argument('--conf-thres', type=float,
                            default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float,
                            default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--max-det', type=int, default=1000,
                            help='maximum number of detections per image')
        parser.add_argument('--classes', nargs='+', type=int,
                            help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true',
                            help='class-agnostic NMS')

        parser.add_argument("--P_REID_CKPT", type=str,
                            default="./ckpt.net64do.mot16.t7")
        parser.add_argument("--V_REID_CKPT", type=str,
                            default="./ckpt.net128do.bitv.t7")
        parser.add_argument("--MAX_DIST", type=float, default=0.2)
        parser.add_argument("--MIN_CONFIDENCE", type=float, default=0.3)
        parser.add_argument("--NMS_MAX_OVERLAP", type=float, default=0.5)
        parser.add_argument("--MAX_IOU_DISTANCE", type=float, default=0.7)
        parser.add_argument("--MAX_AGE", type=int, default=70)
        parser.add_argument("--N_INIT", type=int, default=3)
        parser.add_argument("--NN_BUDGET", type=int, default=100)

        opt = parser.parse_args()

        return opt

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114),
                  auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # only scale down, do not scale up (for better test mAP)
        if not scaleup:
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
            new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / \
                shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)
