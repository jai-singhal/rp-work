import numpy as np
import sys
import cv2
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2
from pprint import pprint
from skimage.filters import unsharp_mask


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
        bgr_t = np.transpose(self.image, (0,2,3,1))
        b, c, h, w = self.image.shape

        sharpness1 = np.zeros(bgr_t.shape, bgr_t.dtype)
        sharpness2 = np.zeros(bgr_t.shape, bgr_t.dtype)

        
        bilateral1 = np.zeros(bgr_t.shape, bgr_t.dtype)
        bilateral2 = np.zeros(bgr_t.shape, bgr_t.dtype)

        pix_up = np.zeros(bgr_t.shape, bgr_t.dtype)
        pix_down = np.zeros(bgr_t.shape, bgr_t.dtype)

        sharpness_u_m1 = np.zeros(bgr_t.shape, bgr_t.dtype)
        sharpness_u_m2 = np.zeros(bgr_t.shape, bgr_t.dtype)

        sharpness_h_p = np.zeros(bgr_t.shape, bgr_t.dtype)
        sharpness_l_p = np.zeros(bgr_t.shape, bgr_t.dtype)

        kernel1 = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        kernel2 = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        
        # High-pass kernel
        kernel4 = np.array([[  0  , -.5 ,    0 ],
                  [-.5 ,   3  , -.5 ],
                  [  0  , -.5 ,    0 ]])
        
        # Low-pass kernel
        kernel5 = np.array([[1 / 9, 1 / 9, 1 / 9],
                  [1 / 9, 1 / 9, 1 / 9],
                  [1 / 9, 1 / 9, 1 / 9]])
        
        for i in range(0,b):
            # sharpness
            sharpness1[i]  = cv2.filter2D(bgr_t[i], -1, kernel1)
            sharpness2[i]  = cv2.filter2D(bgr_t[i], -1, kernel2)

            sharpness_h_p[i]  = cv2.filter2D(bgr_t[i], -1, kernel4)
            sharpness_l_p[i]  = cv2.filter2D(bgr_t[i], -1, kernel5)
            
            sharpness_u_m1[i]  = unsharp_mask(bgr_t[i], radius=5, amount=1, multichannel = True).astype(np.float32)
            sharpness_u_m2[i]  = unsharp_mask(bgr_t[i], radius=5, amount=2, multichannel = True).astype(np.float32)

            # smoothening
            bilateral1[i]    = cv2.bilateralFilter(bgr_t[i], d=5, sigmaColor=1.0, sigmaSpace=5)
            bilateral2[i]   = cv2.bilateralFilter(bgr_t[i], d=5, sigmaColor=0.1, sigmaSpace=5)

            pix_up[i]        = bgr_t[i]*1.05
            pix_down[i]      = bgr_t[i]*0.95

            
        if self.m % 50 == 0:
            (unique, counts) = np.unique(act, return_counts=True)
            frequencies = np.asarray((unique, counts)).T
            pprint(frequencies)
        
        self.m += 1
        

        act_3channel = np.stack([act, act, act],axis=1)

        self.image = np.where(act_3channel == 1, np.transpose(sharpness1, (0,3,1,2)), self.image)
        self.image = np.where(act_3channel == 2, np.transpose(sharpness1, (0,3,1,2)), self.image)

        self.image = np.where(act_3channel == 3, np.transpose(bilateral1, (0,3,1,2)), self.image)
        self.image = np.where(act_3channel == 4, np.transpose(bilateral2, (0,3,1,2)), self.image)

        self.image = np.where(act_3channel == 5, np.transpose(pix_up, (0,3,1,2)), self.image)
        self.image = np.where(act_3channel == 6, np.transpose(pix_down, (0,3,1,2)), self.image)

        self.image = np.where(act_3channel == 7, np.transpose(sharpness_l_p, (0,3,1,2)), self.image)
        self.image = np.where(act_3channel == 8, np.transpose(sharpness_h_p, (0,3,1,2)), self.image)
        
        self.image = np.where(act_3channel == 9, np.transpose(sharpness_u_m1, (0,3,1,2)), self.image)
        self.image = np.where(act_3channel == 10, np.transpose(sharpness_u_m2, (0,3,1,2)), self.image)

        
        self.tensor[:,:self.image.shape[1],:,:] = self.image
        self.tensor[:,-64:,:,:] = inner_state
