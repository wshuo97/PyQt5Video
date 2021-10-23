# from typing import OrderedDict
try:
    from collections import OrderedDict
except ImportError:
    pass
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .model2 import Net64do, Net128do

# model.load_state_dict({k.replace('module.',''):v for k,v in torch.load('myfile.pth').items()})

class Extractor2(object):
    def __init__(self, model_path, classnum, use_cuda=True, device="cuda:0", size=(64, 128)):
        # MOT16
        if classnum == 490:
            self.net = Net64do(num_classes=classnum, reid=True)
        elif classnum == 6:
            self.net = Net128do(num_classes=classnum, reid=True)
        # casco
        elif classnum == 64:
            self.net = Net64do(num_classes=classnum, reid=True)
        elif classnum == 164:
            self.net = Net128do(num_classes=classnum, reid=True)
        self.device = device
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)[
            'net_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = size
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32), size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor2("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)
