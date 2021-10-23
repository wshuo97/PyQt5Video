from .deep_sort import DeepSort
from .deep_sort2 import DeepSort2


__all__ = ['DeepSort', 'build_tracker']


def build_tracker(cfg, use_cuda, device):
    return DeepSort(cfg.REID_CKPT, 751,
                    max_dist=cfg.MAX_DIST, min_confidence=cfg.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.NMS_MAX_OVERLAP, max_iou_distance=cfg.MAX_IOU_DISTANCE,
                    max_age=cfg.MAX_AGE, n_init=cfg.N_INIT, nn_budget=cfg.NN_BUDGET,
                    use_cuda=use_cuda, device=device)


def build_tracker2(ckpt_path, classnum, cfg, use_cuda, device):
    return DeepSort(ckpt_path, classnum,
                    max_dist=cfg.MAX_DIST, min_confidence=cfg.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.NMS_MAX_OVERLAP, max_iou_distance=cfg.MAX_IOU_DISTANCE,
                    max_age=cfg.MAX_AGE, n_init=cfg.N_INIT, nn_budget=cfg.NN_BUDGET,
                    use_cuda=use_cuda, device=device)


def build_tracker3(ckpt_path, classnum, cfg, use_cuda, device, size):
    return DeepSort2(ckpt_path, classnum,
                    max_dist=cfg.MAX_DIST, min_confidence=cfg.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.NMS_MAX_OVERLAP, max_iou_distance=cfg.MAX_IOU_DISTANCE,
                    max_age=cfg.MAX_AGE, n_init=cfg.N_INIT, nn_budget=cfg.NN_BUDGET,
                    use_cuda=use_cuda, device=device, size=size)
