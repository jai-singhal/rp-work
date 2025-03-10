import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
import numpy as np
import math
import cv2
from chainer.links.caffe import CaffeFunction
import chainerrl
from chainerrl.agents import a3c

class DilatedConvBlock(chainer.Chain):

    def __init__(self, d_factor):
        super(DilatedConvBlock, self).__init__(
            diconv=L.DilatedConvolution2D( in_channels=64, out_channels=64, ksize=3, stride=1, pad=d_factor, dilate=d_factor, nobias=False),
            #bn=L.BatchNormalization(64)
        )

        self.train = True

    def __call__(self, x):
        h = F.relu(self.diconv(x))
        #h = F.relu(self.bn(self.diconv(x)))
        return h


class MyFcn(chainer.Chain, a3c.A3CModel):
 
    def __init__(self, n_actions):
        w = chainer.initializers.HeNormal()

        super(MyFcn, self).__init__(
            conv1=L.Convolution2D( 3, 64, 3, stride=1, pad=1, nobias=False),
            diconv2=DilatedConvBlock(2),
            diconv3=DilatedConvBlock(3),
            diconv4=DilatedConvBlock(4),
            diconv5_pi=DilatedConvBlock(3),
            diconv6_pi=DilatedConvBlock(2),
            conv7_Wz=L.Convolution2D( 64, 64, 3, stride=1, pad=1, nobias=True),
            conv7_Uz=L.Convolution2D( 64, 64, 3, stride=1, pad=1, nobias=True),
            conv7_Wr=L.Convolution2D( 64, 64, 3, stride=1, pad=1, nobias=True),
            conv7_Ur=L.Convolution2D( 64, 64, 3, stride=1, pad=1, nobias=True),
            conv7_W=L.Convolution2D( 64, 64, 3, stride=1, pad=1, nobias=True),
            conv7_U=L.Convolution2D( 64, 64, 3, stride=1, pad=1, nobias=True),
            conv8_pi=chainerrl.policies.SoftmaxPolicy(L.Convolution2D( 64, n_actions, 3, stride=1, pad=1, nobias=False)),
            diconv5_V=DilatedConvBlock(3),
            diconv6_V=DilatedConvBlock(2),
            conv7_V=L.Convolution2D( 64, 1, 3, stride=1, pad=1, nobias=False),
            conv_R=L.Convolution2D( 1, 1, 33, stride=1, pad=16, nobias=True),
        )
        self.train = True
 
    def pi_and_v(self, x):
         
        h = F.relu(self.conv1(x[:,0:3,:,:]))
        h = self.diconv2(h)
        h = self.diconv3(h)
        h = self.diconv4(h)
        h_pi = self.diconv5_pi(h)
        x_t = self.diconv6_pi(h_pi)
        h_t1 = x[:,-64:,:,:]
        z_t = F.sigmoid(self.conv7_Wz(x_t)+self.conv7_Uz(h_t1))
        r_t = F.sigmoid(self.conv7_Wr(x_t)+self.conv7_Ur(h_t1))
        h_tilde_t = F.tanh(self.conv7_W(x_t)+self.conv7_U(r_t*h_t1))
        h_t = (1-z_t)*h_t1+z_t*h_tilde_t
        pout = self.conv8_pi(h_t)

        h_V = self.diconv5_V(h)
        h_V = self.diconv6_V(h_V)
        vout = self.conv7_V(h_V)
       
        return pout, vout, h_t

    def conv_smooth(self, x):
        x = self.conv_R(x)
        return x
