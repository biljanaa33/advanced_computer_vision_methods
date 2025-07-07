import numpy as np
import cv2
import matplotlib . pyplot as plt

from ex2_utils import Tracker, get_patch, create_epanechnik_kernel, extract_histogram, backproject_histogram


class MSParams():

    def __init__(self):
        self.bins = 16
        self.eps = 1e-1
        self.sigma = 1
        self.max_iter = 20
        self.alpha = 0.0
        self.bg_neighborhood = 3
        self.bg_flag = False
        self.color_space = None

color_map = {

            "RGB": cv2.COLOR_BGR2RGB,
            "HSV": cv2.COLOR_BGR2HSV,
            "YCbCr": cv2.COLOR_BGR2YCrCb,
            "Lab": cv2.COLOR_BGR2Lab, 
         }


class MeanShiftTracker(Tracker):

    def initialize(self, image, region):

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

       
        if self.parameters.color_space in color_map:
            image = cv2.cvtColor(image, color_map[self.parameters.color_space])
 

        top_x = int(region[0])
        top_y = int(region[1])
        width = int(region[2])
        height = int(region[3])
        # enusre odd size
        if width % 2 == 0: width -= 1
        if height % 2 == 0: height -= 1

        self.size = [width, height]
        self.position = [int(top_x + width/2), int(top_y + height/2)]

        
        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.parameters.sigma)
        
        if self.parameters.bg_flag == True: 
            self.bg_kernel = create_epanechnik_kernel(self.size[0] * self.parameters.bg_neighborhood, 
                                                  self.size[1] * self.parameters.bg_neighborhood, 
                                                  self.parameters.sigma)
   

        patch,mask = get_patch(image, self.position, self.size)
        templ_kernel = mask * self.kernel
        self.q_hist = extract_histogram(patch, self.parameters.bins, templ_kernel)
        
        if self.parameters.bg_flag == True:
           self.c = self.background_correction(image)
        else: 
            self.c = 1

        self.q_hist = self.c * self.q_hist
        self.q_hist = self.q_hist / np.sum(self.q_hist)

        # use for mean sift calc 
        x_vals = np.arange(-width//2 + 1, width//2 + 1)
        y_vals = np.arange(-height//2 + 1, height//2 + 1)
        [self.x_pos, self.y_pos] = np.meshgrid(x_vals, y_vals)

        

    def track(self, image):
        
        if self.parameters.color_space in color_map:
            image = cv2.cvtColor(image, color_map[self.parameters.color_space])

        for _ in range(self.parameters.max_iter):
            
            patch,mask = get_patch(image, self.position, self.size)

     
            patch_kernel = self.kernel * mask
            p_hist = extract_histogram(patch, self.parameters.bins, patch_kernel) # * self.c dont use this better results 
            p_hist = p_hist/(np.sum(p_hist) + 1e-7)

            v_hist = np.sqrt( self.q_hist / (p_hist + 1e-7))
            backproject = backproject_histogram(patch, v_hist , self.parameters.bins) 
            backproject = backproject * self.kernel
            backproject = backproject/(np.sum(backproject) + 1e-7)


            x_shift = np.sum(self.x_pos*backproject) / (np.sum(backproject) + 1e-7)
            y_shift = np.sum(self.y_pos*backproject) / (np.sum(backproject) + 1e-7)

            if np.linalg.norm(np.array(x_shift, y_shift)) < self.parameters.eps:
                # print(i)
                break

            self.position[0] += x_shift
            self.position[1] += y_shift


        upd_hist = extract_histogram(patch, self.parameters.bins, self.kernel) * self.c
        upd_hist = upd_hist/np.sum(upd_hist)
        self.q_hist = (1-self.parameters.alpha) * self.q_hist + self.parameters.alpha * upd_hist
        
        
        return [self.position[0] - self.size[0]//2, self.position[1] - self.size[1]//2, self.size[0], self.size[1]]
    
    def background_correction(self, image):

        # print(np.array(self.size) * self.parameters.bg_neighborhood)
        bg_patch, _ = get_patch(image, self.position, np.array(self.size) * self.parameters.bg_neighborhood)
        bg_hist = extract_histogram(bg_patch, self.parameters.bins, self.bg_kernel)
        #bg_hist -= self.q_hist not good, introduces negative values
        #bg_hist = np.maximum(bg_hist, 0)

        o_hat = min([val for val in bg_hist if val > 0])
        o_hat = np.array([o_hat] * len(bg_hist))
        c_hist = np.minimum(o_hat / (bg_hist + 1e-9), 1)
        return c_hist
        

 