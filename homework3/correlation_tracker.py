import numpy as np
import cv2
from ex2_utils import Tracker, get_patch
from ex3_utils import create_cosine_window, create_gauss_peak
import os
import time
from utils.tracker import Tracker
# from ex2_utils import Tracker

class CorrTracker(Tracker):

    def __init__(self):

        super().__init__()
        self.alpha = float(os.environ.get("ALPHA", 0.2))
        self.sigma = float(os.environ.get("SIGMA", 2.0))
        self.lmbd = float(os.environ.get("LMBDA", 500))
        self.enlarge_factor = float(os.environ.get("ENLFCT", 1.0))
                
        self.position = None
        self.size = None
        self.patch_size = None
        self.cos_window = None
        self.G = None
        self.H = None

        self.time_flag = True
        self.init_times = []
        self.track_times = []

    def initialize(self, frame, region):

        # print(self.alpha, self.sigma)
        if self.time_flag: 
            start_time = time.time()

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        top_x = int(region[0])
        top_y = int(region[1])
        width = int(region[2])
        height = int(region[3])

        # ensure odd dimensions
        if width % 2 == 0: width -= 1
        if height % 2 == 0: height -= 1

        self.size = [width, height]
        self.patch_size = [int(width * self.enlarge_factor), int(height * self.enlarge_factor)]

        if self.patch_size[0] % 2 == 0: self.patch_size[0] -= 1
        if self.patch_size[1] % 2 == 0: self.patch_size[1] -= 1

        self.position = [int(top_x + width / 2), int(top_y + height / 2)]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        patch, mask = get_patch(gray, self.position, self.patch_size)
        # print(patch.shape)
        self.cos_window = create_cosine_window((patch.shape[1], patch.shape[0]))
        F = self.preprocess_patch(patch * mask)

        self.G = create_gauss_peak((patch.shape[1], patch.shape[0]), sigma=self.sigma)
        self.H = self.create_filter(F, self.G)

        if self.time_flag:
            self.init_times.append(time.time() - start_time)

    def track(self, frame):

        if self.time_flag:
            start_time = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        patch, mask = get_patch(gray, self.position, self.patch_size)

        F = self.preprocess_patch(patch * mask)
        F_hat = np.fft.fft2(F)
        response = np.fft.ifft2(F_hat * self.H)

        dy, dx = np.unravel_index(np.argmax(np.abs(response)), response.shape)

        # handle circular shift
        if dx > response.shape[1] // 2:
            dx -= response.shape[1]
        if dy > response.shape[0] // 2:
            dy -= response.shape[0]

        # update
        self.position[0] += dx
        self.position[1] += dy

        patch_new, mask_new = get_patch(gray, self.position, self.patch_size)

        F_new = self.preprocess_patch(patch_new * mask_new)
        H_new = self.create_filter(F_new, self.G)

        # update the filter
        self.H = (1 - self.alpha) * self.H + self.alpha * H_new

        if self.time_flag:
            self.track_times.append(time.time() - start_time)

        return [self.position[0] - self.size[0] // 2, self.position[1] - self.size[1] // 2, self.size[0], self.size[1]]
    
    
    def preprocess_patch(self, patch): 
        
        patch = (np.float32(patch) - np.mean(patch)) / (np.std(patch) + 1e-7)
        return patch * self.cos_window

    def create_filter(self, F, G):
        F_hat = np.fft.fft2(np.float32(F))
        G_hat = np.fft.fft2(np.float32(G))
        H = (G_hat * np.conj(F_hat)) / (F_hat * np.conj(F_hat) + self.lmbd)
        return H

    def name(self):
        return "tracker_cf"
    
    
