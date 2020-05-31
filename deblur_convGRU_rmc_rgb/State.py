import numpy as np
import sys
import cv2
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2
from pprint import pprint

class State():
    def __init__(self, size):
        self.image = np.zeros(size,dtype=np.float32)
        self.m = 0
        
    def reset(self, n):
        self.image = n
        size = self.image.shape
        prev_state = np.zeros((size[0],64,size[2],size[3]),dtype=np.float32)
        self.tensor = np.concatenate((self.image, prev_state), axis=1)

    def set(self, x):
        self.image = x
        self.tensor[:,:self.image.shape[1],:,:] = self.image

        
    def step(self, act, inner_state):
        # gaussian = np.zeros(self.image.shape, self.image.dtype)
        # median = np.zeros(self.image.shape, self.image.dtype)
        sharpness1 = np.zeros(self.image.shape, self.image.dtype)
        sharpness2 = np.zeros(self.image.shape, self.image.dtype)
        u_wiener = np.zeros(self.image.shape, self.image.dtype)
        wiener_up = np.zeros(self.image.shape, self.image.dtype)
        wiener_down = np.zeros(self.image.shape, self.image.dtype)
        wiener_deconv = np.zeros(self.image.shape, self.image.dtype)
        bilateral = np.zeros(self.image.shape, self.image.dtype)
#         box = np.zeros(self.image.shape, self.image.dtype)
#         richardson = np.zeros(self.image.shape, self.image.dtype)

        
        b, c, h, w = self.image.shape
        psf = np.ones((5, 5)) / 25
        kernel1 = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        kernel2 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

#         for i in range(0,b):
#             cv2.imwrite('./temp/' + str(i) + '_input.png', (self.image[i,0]*255).astype(np.uint8))
        
        bgrt = (self.image*255).astype(np.uint8)
        for i in range(0,b):
            # sharpness
            sharpness1[i, 0] = cv2.filter2D(bgrt[i, 0], -1, kernel1).astype(np.uint8)
            sharpness2[i, 0] = cv2.filter2D(bgrt[i, 0], -1, kernel2).astype(np.uint8)
            
            bilateral[i,0] = cv2.bilateralFilter(bgrt[i,0], d=5, sigmaColor=0.1, sigmaSpace=5).astype(np.uint8)

            temp, _ = restoration.unsupervised_wiener(bgrt[i, 0], psf, clip = False)
            u_wiener[i, 0] = temp.astype(np.uint8)
        
            wiener_up[i,0] = restoration.wiener(bgrt[i, 0], psf, clip = False, balance = 1.15).astype(np.uint8)
            wiener_down[i,0] = restoration.wiener(bgrt[i, 0], psf, clip = False, balance = 0.95).astype(np.uint8)
    
        if self.m % 50 == 0:
            (unique, counts) = np.unique(act, return_counts=True)
            frequencies = np.asarray((unique, counts)).T
            pprint(frequencies)
        
        self.m += 1
#         print(self.image, sharpness1)
        
        act_3channel = np.stack([act],axis=1)

        bgrt = np.where(act_3channel == 1, sharpness1, bgrt)
        bgrt = np.where(act_3channel == 2, bilateral, bgrt)
        bgrt = np.where(act_3channel == 3, u_wiener, bgrt)
        bgrt = np.where(act_3channel == 4, wiener_up, bgrt)
        bgrt = np.where(act_3channel == 5, wiener_down, bgrt)
        bgrt = np.where(act_3channel == 6, sharpness2, bgrt)
        
        self.image = (bgrt/255).astype(np.float32)
#         for i in range(0,b):
#             cv2.imwrite('./temp/' + str(i) + '_output.png', (self.image[i,0]*255).astype(np.uint8))
        
        self.tensor[:,:self.image.shape[1],:,:] = self.image
        self.tensor[:,-64:,:,:] = inner_state
