import cv2 as cv
import numpy as np


class Featurizer:
    def __init__(self, nfeatures: int = 100):
        self.sift = cv.xfeatures2d.SIFT_create(nfeatures=nfeatures)

    def get_features(self, path: str):
        img = cv.imread(path)
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        keypoints, desc = self.sift.detectAndCompute(gray_img, None)
        return np.matrix(desc)
