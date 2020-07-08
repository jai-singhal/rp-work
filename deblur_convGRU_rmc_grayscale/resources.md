## Resources

-https://www.taylorpetrick.com/blog/post/convolution-part3 

- Network and kernel estimation: Learning to Deblur
	https://arxiv.org/pdf/1406.7444.pdf

- Base Paper: Learning Deep CNN Denoiser Prior for Image Restoration
	https://arxiv.org/pdf/1704.03264.pdf

- Filters and deblurring
	https://www.researchgate.net/publication/322589726_Deblurring_techniques_for_natural_images_-_A_comparative_and_comprehensive_study

- A3C implementation in chainerRL
	https://github.com/chainer/chainerrl/blob/master/chainerrl/agents/a3c.py


- Using YCbCr color space ::
Super-resolving a Single Blurry Image Through Blind Deblurring Using ADMM
Xiangrong Zeng, Yan Liu, Jun Fan, Qizi Huangpeng, Jing Feng, Jinlun Zhou, Maojun Zhang



## Filters:

# Non-Blind Deconvolution 
1. Weiner Filter
2. Unsupervised Wiener
3. Inverse Filter
4. Richardson-Lucy algorithm 

# shaprness
5. Sharpness kerel 1: 
	- [[0, -1, 0], [-1, 5,-1], [0, -1, 0]]
	
6.Sharpness kerel 2:
	- [[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]

# Smoothness
7. GaussianBlur(bgr_t[i], ksize=(3,3), sigmaX=0.5)
8. bilateralFilter(bgr_t[i], d=3, sigmaColor=0.1, sigmaSpace=5)
9. boxFilter(bgr_t[i], -1, (3,3), normalize = True)
10. medianBlur(bgr_t[i], 3)

### Ref:
- https://scikit-image.org/docs/dev/auto_examples/filters/plot_restoration.html
- https://github.com/scikit-image/scikit-image/blob/master/skimage/restoration/deconvolution.py
- http://www.robots.ox.ac.uk/~az/lectures/ia/lect3.pdf


https://sandipanweb.wordpress.com/2018/07/30/some-image-processing-problems/

https://www.clear.rice.edu/elec431/projects95/lords/wolf.html
https://www.diva-portal.org/smash/get/diva2:330667/FULLTEXT02.pdf