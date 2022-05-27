import cv2
import mediapipe as mp
import numpy as np
from typing import Literal


class SelfiSegmentation:

    def __init__(self, model: Literal[0, 1] = 1):
        """
        :param model: model type --> (0 or 1). 0 is general 1 is landscape(faster)
        """
        self.model = model
        self.mpDraw = mp.solutions.drawing_utils
        self.mpSelfieSegmentation = mp.solutions.selfie_segmentation
        self.selfieSegmentation = self.mpSelfieSegmentation.SelfieSegmentation(self.model)

    def virtualBG(self, img: np.dtype, virtualBg=None, threshold: float = 0.1):
        """
        :param img: image to remove background from
        :param virtualBg: BackGround Image
        :param threshold: higher = more cut, lower = less cut
        :return:
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.selfieSegmentation.process(imgRGB)
        binary_mask = results.segmentation_mask > threshold
        condition = np.dstack((binary_mask, binary_mask, binary_mask))
        if virtualBg is None:
            virtualBg = cv2.blur(img, (25, 25))
            _imgBg = np.zeros(img.shape, dtype=np.uint8)
            _imgBg[:] = virtualBg
            imgOut = np.where(condition, img, _imgBg)
        else:
            imgOut = np.where(condition, img, virtualBg)
        return imgOut



if __name__ == "__main__":
    pass
