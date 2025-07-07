from ex4_utils import sample_gauss
import numpy as np 
import sympy as sp
from matplotlib import pyplot as plt
import math
from ex2_utils import create_epanechnik_kernel, get_patch, extract_histogram
from utils.tracker import Tracker
import cv2

np.random.seed(42)

class ParticleFilterTracker(Tracker):

    def __init__(self, model = "nca", color = "bgr"):

        self.model = model
        self.color = color

    def initialize(self, frame, region):

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        top_x = int(region[0])
        top_y = int(region[1])
        width = int(region[2])
        height = int(region[3])

        if width % 2 == 0: width -= 1
        if height % 2 == 0: height -= 1

        center_x = int(top_x + width / 2)
        center_y = int(top_y + height / 2)

        # self.N = 100 # number of particles
        self.N = 100
        self.frame_h, self.frame_w = frame.shape[:2]
        self.nbins = 16
        # ensure odd dimensions

        self.size = [width, height]

        frame = self.convert_colorspace(frame)
     
        # set visual model
        self.kernel = create_epanechnik_kernel(width, height, sigma=2.0)
        patch, mask = get_patch(frame, (center_x, center_y), (self.size[0], self.size[1]))
        weights = self.kernel * mask
        self.h_ref = extract_histogram(patch, nbins=self.nbins, weights=weights)
        self.h_ref = self.h_ref / np.sum(self.h_ref)
        self.weights = np.ones(self.N) / self.N
        self.q = 0.1 * min(width, height)
        self.sigma = 0.1 # perv sigma 
        self.alpha = 0.0005 # self.alpha = 0.0005

        if self.model == "ncv":
            self.A = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)

            self.Q = self.q * np.array([
                [1/3, 0, 1/2, 0],
                [0, 1/3, 0, 1/2],
                [1/2, 0, 1, 0],
                [0, 1/2, 0, 1]
            ], dtype=np.float32)

            mu = np.zeros(4, dtype=np.float32)
            mu[0] = center_x
            mu[1] = center_y

        elif self.model == "nca":
            self.A = np.array([
                [1, 0, 1, 0, 0.5, 0],
                [0, 1, 0, 1, 0, 0.5],
                [0, 0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ], dtype=np.float32)

            self.Q = self.q * np.array([
                [1/20, 0, 1/8, 0, 1/6, 0],
                [0, 1/20, 0, 1/8, 0, 1/6],
                [1/8, 0, 1/3, 0, 1/2, 0],
                [0, 1/8, 0, 1/3, 0, 1/2],
                [1/6, 0, 1/2, 0, 1, 0],
                [0, 1/6, 0, 1/2, 0, 1]
            ], dtype=np.float32)

            mu = np.zeros(6, dtype=np.float32)
            mu[0] = center_x
            mu[1] = center_y

        else:
            self.A = np.eye(2, dtype=np.float32)
            self.Q = self.q * np.eye(2, dtype=np.float32)

            mu = np.zeros(2, dtype=np.float32)
            mu[0] = center_x
            mu[1] = center_y


        self.particles = sample_gauss(mu, self.Q, self.N)


    def track(self, frame):
        # print("tuka")

        frame = self.convert_colorspace(frame)
        self.resample_particles() #  1. resample new particles
        self.predict_particles()  # 2. move each particle using the dynamic model
        self.update_weights(frame)  # 3. update weightes based on visual model similarity
        estimate = self.estimate_state()  # 4. compute new state as a weighted sum of particle states

        cx, cy = estimate
        patch, mask = get_patch(frame, (cx, cy), (self.size[0], self.size[1]))
        weights = self.kernel * mask

        new_h = extract_histogram(patch, nbins=self.nbins, weights=weights)
        new_h = new_h / np.sum(new_h) if np.sum(new_h) > 0 else np.ones_like(new_h) / len(new_h)
        self.h_ref = (1 - self.alpha) * self.h_ref + self.alpha * new_h

        return [int(cx-self.size[0]/2), int(cy - self.size[1]/2), int(self.size[0]), int(self.size[1])]


    def resample_particles(self): 

        weights_norm = self.weights / (np.sum(self.weights) + 1e-100)
        weights_cumsum = np.cumsum(weights_norm)
        rand_samples = np.random.rand(self.N, 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsum)
        self.particles = self.particles[sampled_idxs.flatten(), :] 

    def predict_particles(self):
        noise = sample_gauss(np.zeros(self.A.shape[0]), self.Q, self.N)
        self.particles = (self.particles @ self.A.T) + noise

        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, self.frame_w - 1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, self.frame_h - 1)


    def estimate_state(self):
            
            estimated_state = np.average(self.particles, axis=0, weights=self.weights)
            return estimated_state[:2]

    def update_weights(self, frame):
        
        for i in range(self.N):

            if self.model == "ncv" : x, y, _, _ = self.particles[i]
            elif self.model == "nca": x, y, _, _, _, _ = self.particles[i]
            else: x, y  = self.particles[i]

            patch, mask = get_patch(frame, (x, y), (self.size[0], self.size[1]))
            weights = self.kernel * mask
            hist = extract_histogram(patch, nbins=self.nbins, weights=weights)
            hist = hist / np.sum(hist) if np.sum(hist) > 0 else np.ones_like(hist) / len(hist)
            dist = np.sqrt(np.sum((np.sqrt(hist) - np.sqrt(self.h_ref))**2)) / np.sqrt(2)
            self.weights[i] = np.exp(-dist**2 / (2 * self.sigma**2))

        self.weights /= np.sum(self.weights)

    def convert_colorspace(self, frame):

        if self.color == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self.color == "hsv":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif self.color == "lab":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        elif self.color == "ycrcb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        return frame

    def name(self):
        return "tracker_particle"
 


