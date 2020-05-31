import numpy as np
import sys
import cv2
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2
from pprint import pprint

class State():
    def __init__(self, size, move_range):
        self.image = np.zeros(size,dtype=np.float32)
        self.move_range = move_range
        self.m = 0
        
    def reset(self, x, n):
        self.image = x
        size = self.image.shape
        prev_state = np.zeros((size[0],64,size[2],size[3]),dtype=np.float32)
        self.tensor = np.concatenate((self.image, prev_state), axis=1)

    def set(self, x):
        self.image = x
        self.tensor[:,:self.image.shape[1],:,:] = self.image

        
    def step(self, act, inner_state):
        gaussian = np.zeros(self.image.shape, self.image.dtype)
        median = np.zeros(self.image.shape, self.image.dtype)
        sharpness1 = np.zeros(self.image.shape, self.image.dtype)
        sharpness2 = np.zeros(self.image.shape, self.image.dtype)
        u_wiener = np.zeros(self.image.shape, self.image.dtype)
        wiener_up = np.zeros(self.image.shape, self.image.dtype)
        wiener_down = np.zeros(self.image.shape, self.image.dtype)
        wiener_deconv = np.zeros(self.image.shape, self.image.dtype)

        b, c, h, w = self.image.shape
        psf = np.ones((5, 5)) / 25
        kernel1 = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        kernel2 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

        for i in range(0,b):
            # sharpness
            sharpness1[i, 0] = cv2.filter2D(self.image[i, 0], -1, kernel1)
            sharpness2[i, 0] = cv2.filter2D(self.image[i, 0], -1, kernel2)
            
            gaussian[i,0] = cv2.GaussianBlur(self.image[i,0], ksize=(5,5), sigmaX=0.5)
            median[i,0] = cv2.medianBlur(self.image[i,0], ksize=5)

            u_wiener[i,0], _ = restoration.unsupervised_wiener(self.image[i, 0], psf, clip = False)
            wiener_up[i,0] = restoration.wiener(self.image[i, 0], psf, clip = False, balance = 1.05)
            wiener_down[i,0] = restoration.wiener(self.image[i, 0], psf, clip = False, balance = 0.95)
    
        if self.m % 5 == 0:
            (unique, counts) = np.unique(act, return_counts=True)
            frequencies = np.asarray((unique, counts)).T
            pprint(frequencies)
        
        self.m += 1
        
        act_3channel = np.stack([act],axis=1)
        self.image = np.where(act_3channel == 1, sharpness1, self.image)
        self.image = np.where(act_3channel == 2, median, self.image)
        self.image = np.where(act_3channel == 3, u_wiener, self.image)
        self.image = np.where(act_3channel == 4, gaussian, self.image)
        self.image = np.where(act_3channel == 5, wiener_up, self.image)
        self.image = np.where(act_3channel == 6, wiener_down, self.image)
        self.image = np.where(act_3channel == 7, sharpness2, self.image)

        self.tensor[:,:self.image.shape[1],:,:] = self.image
        self.tensor[:,-64:,:,:] = inner_state
