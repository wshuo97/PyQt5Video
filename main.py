import argparse
from os import confstr_names
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import os

from utils.datasets import LoadImages
from utils.plots import colors, plot_one_box, plot_targets_txt
from utils.torch_utils import time_synchronized
from deep_sort import build_tracker3
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, increment_path, non_max_suppression, scale_coords


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    elif data_type == "darklabel":
        save_format = "{frame},{cname},{id},{x1},{y1},{w},{h}\n"
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, cls_name, tlwh, track_id in results:
            if data_type == 'kitti':
                frame_id -= 1
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            line = save_format.format(
                frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, cname=cls_name)
            f.write(line)


def get_model(opt):
    source = opt.source
    weights = opt.weights
    pweight, vweight = opt.P_REID_CKPT, opt.V_REID_CKPT
    pnum, vnum = opt.pnum, opt.vnum
    use_cuda = torch.cuda.is_available()
    device = opt.device if use_cuda else "cpu"
    imgsz = 640
    stride = 32

    # detector
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    model.half()  # to FP16

    # tracker
    ptracker = build_tracker3(
        pweight, pnum, opt, use_cuda, device, (64, 128))
    vtracker = build_tracker3(
        vweight, vnum, opt, use_cuda, device, (128, 128))

    # dataset
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    return model, ptracker, vtracker, dataset


def detect(opt):
    save_txt = opt.save_txt
    device = opt.device
    save_dir = increment_path(
        Path(opt.project) / opt.name, exist_ok=opt.exist_ok) if opt.save_dir == "null" \
        else increment_path(Path(opt.save_dir), exist_ok=opt.exist_ok)
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
    #                                                       exist_ok=True)  # make dir

    save_results_path = save_dir/"results.txt"

    results = []

    detector, ptracker, vtracker, dataset = get_model(opt)

    t0 = time.time()

    vid_path, vid_writer = None, None

    cnames = ["person", "vehicle"]

    # input()
    frameid = 0
    """
        frame: frame size of ./data/casco.avi, (540, 960, 3)
        img  : RGB model image, (3, 384, 640)
        im0s : BGR model image, (540, 960, 3)
    """
    for path, img, im0s, vid_cap in dataset:

        frameid = frameid+1

        img = torch.from_numpy(img).to(device)
        img = img.half()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = detector(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)

        bbox_xywh = []
        cls_ids = None
        cls_conf = None
        for i, det in enumerate(pred):
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

        poutputs = []
        voutputs = []

        if len(bbox_xywh) > 0:
            pmask = (cls_ids == 0)
            pbbox_xywh = bbox_xywh[pmask]
            pcls_conf = cls_conf[pmask]
            pcls_ids = cls_ids[pmask]
            poutputs = ptracker.update(bbox_xywh=pbbox_xywh, classid=pcls_ids,
                                       confidences=pcls_conf, ori_img=im0s)
            vmask = (cls_ids == 1) + (cls_ids == 2) + (cls_ids == 3) + \
                (cls_ids == 4) + (cls_ids == 5) + \
                (cls_ids == 6) + (cls_ids == 7)
            vbbox_xywh = bbox_xywh[vmask]
            vcls_conf = cls_conf[vmask]
            vcls_ids = cls_ids[vmask]
            voutputs = vtracker.update(bbox_xywh=vbbox_xywh, classid=vcls_ids,
                                       confidences=vcls_conf, ori_img=im0s)

        t2 = time_synchronized()

        p, s, im0 = path, '', im0s.copy()
        for i, det in enumerate([poutputs, voutputs]):  # detections per image

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                for *xyxy, idx in reversed(det):
                    labelid = f'{idx}'
                    results.append(
                        (frameid-1, cnames[i], ptracker._xyxy_to_tlwh(xyxy), idx))
                    plot_one_box(xyxy, im0, label=labelid, color=colors(
                        idx % 255, True), line_thickness=opt.line_thickness)
            if i == 1:
                print(f'{s}Done. ({t2 - t1:.3f}s)')

        if vid_path != save_path:  # new video
            vid_path = save_path
            if vid_cap:  # video
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # print("\n***", save_path, fps, w, h, "***")
            vid_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        vid_writer.write(im0)

    write_results(save_results_path, results, "mot")

    print(f'Done. ({time.time() - t0:.3f}s)')
    return f"Results saved to {save_dir}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # detector config
    parser.add_argument('--weights', type=str, default='./weights/yolov5ft.pt')
    parser.add_argument('--source', type=str, default='./data/casco/casco.avi')
    parser.add_argument('--detpath', type=str, default='')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--save-txt', type=bool, default=True)
    parser.add_argument('--save-conf', action='store_true')
    parser.add_argument('--save-dir', type=str, default="null")
    parser.add_argument('--project', default='runs/detect')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--line-thickness', default=3)
    parser.add_argument('--pnum', type=int, default=64)
    parser.add_argument('--vnum', type=int, default=164)
    parser.add_argument('--classes', nargs='+', type=int)
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--update', action='store_true',)
    parser.add_argument('--hide-labels', default=False)
    parser.add_argument('--hide-conf', default=False)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--max-det', type=int, default=1000)
    parser.add_argument('--tracklets', action='store_true')

    parser.add_argument("--P_REID_CKPT", type=str,
                        default="./weights/ckpt.net64do.t7")
    parser.add_argument("--V_REID_CKPT", type=str,
                        default="./weights/ckpt.net128do.t7")
    parser.add_argument("--MAX_DIST", type=float, default=0.2)
    parser.add_argument("--MIN_CONFIDENCE", type=float, default=0.3)
    parser.add_argument("--NMS_MAX_OVERLAP", type=float, default=0.5)
    parser.add_argument("--MAX_IOU_DISTANCE", type=float, default=0.7)
    parser.add_argument("--MAX_AGE", type=int, default=70)
    parser.add_argument("--N_INIT", type=int, default=3)
    parser.add_argument("--NN_BUDGET", type=int, default=100)

    opt = parser.parse_args()

    print("-"*10)
    for key, value in vars(opt).items():
        print(f"{key:17} : {value}")
    print("-"*10)

    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    rets = []

    with torch.no_grad():
        rets.append(detect(opt=opt))
    for i in rets:
        print(i)
