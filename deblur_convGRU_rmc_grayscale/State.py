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
        sharpness1 = np.zeros(self.image.shape, self.image.dtype)
        sharpness2 = np.zeros(self.image.shape, self.image.dtype)
        bilateral1 = np.zeros(self.image.shape, self.image.dtype)
        bilateral2 = np.zeros(self.image.shape, self.image.dtype) 
        pix_up = np.zeros(self.image.shape, self.image.dtype)
        pix_down = np.zeros(self.image.shape, self.image.dtype)
        
        sharpness_u_m1 = np.zeros(self.image.shape, self.image.dtype)
        sharpness_u_m2 = np.zeros(self.image.shape, self.image.dtype)

        sharpness_h_p = np.zeros(self.image.shape, self.image.dtype)
        sharpness_l_p = np.zeros(self.image.shape, self.image.dtype)

        b, c, h, w = self.image.shape
        psf = np.ones((3, 3)) / 9
        kernel1 = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        kerenl_1_3 = np.array([[0, -3, 0], [-3, 13,-3], [0, -3, 0]])
        kerenl_1_5 = np.array([[0, -5, 0], [-5, 21,-5], [0, -5, 0]])
        
        kernel2 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        

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
            sharpness1[i, 0]  = cv2.filter2D(self.image[i, 0], -1, kernel1)
            sharpness2[i, 0]  = cv2.filter2D(self.image[i, 0], -1, kernel2)
            
            sharpness_u_m1[i, 0]  = unsharp_mask(self.image[i, 0], radius=5, amount=1)
            sharpness_u_m2[i, 0]  = unsharp_mask(self.image[i, 0], radius=5, amount=2)

            
            sharpness_h_p[i, 0]  = cv2.filter2D(self.image[i, 0], -1, kernel4)
            sharpness_l_p[i, 0]  = cv2.filter2D(self.image[i, 0], -1, kernel5)

            #smoothening
            bilateral1[i,0]    = cv2.bilateralFilter(self.image[i, 0], d=5, sigmaColor=0.1, sigmaSpace=5)
            bilateral2[i,0]    = cv2.bilateralFilter(self.image[i, 0], d=5, sigmaColor=1.0, sigmaSpace=5)

            #brightness
            pix_up[i,0]        = self.image[i, 0]*1.05
            pix_down[i,0]      = self.image[i, 0]*0.95

            
        if self.m % 100 == 0:
            (unique, counts) = np.unique(act, return_counts=True)
            frequencies = np.asarray((unique, counts)).T
            pprint(frequencies)
        
        self.m += 1
        

        act_3channel = np.stack([act],axis=1)

        self.image = np.where(act_3channel == 1, sharpness1, self.image)
        self.image = np.where(act_3channel == 2, bilateral1, self.image)
        self.image = np.where(act_3channel == 3, bilateral2, self.image)
        self.image = np.where(act_3channel == 4, sharpness2, self.image)

        self.image = np.where(act_3channel == 5, pix_up, self.image)
        self.image = np.where(act_3channel == 6, pix_down, self.image)

        self.image = np.where(act_3channel == 7, sharpness_u_m1, self.image)
        self.image = np.where(act_3channel == 8, sharpness_u_m2, self.image)

        self.image = np.where(act_3channel == 9, sharpness_h_p, self.image)
        self.image = np.where(act_3channel == 10, sharpness_l_p, self.image)

        
        self.tensor[:,:self.image.shape[1],:,:] = self.image
        self.tensor[:,-64:,:,:] = inner_state
