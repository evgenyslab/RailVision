#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Development workspace
"""
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob 
from moviepy.editor import VideoFileClip

cap = cv2.VideoCapture('video_files/skytrain.mp4')
ret, I = cap.read()
# sample some 10 stills from the video:

nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)

interval = nframes/10

for i in range(0,10):
    cap.set(cv2.CAP_PROP_POS_FRAMES,i*interval+30)
    ret, image = cap.read()
    cv2.imwrite('frame_{:02d}.png'.format(i), image)



filename = 'video_files/skytrain.mp4'
clip = VideoFileClip(filename)
clip1 = clip.cutout(0, 4)
clip1 = clip1.cutout(12,clip1.duration)
clip1.write_videofile("test.mp4")


"""
Notes
[1280 720]
vetical cutoff:
    345 pixels
Horizontal Window:
    [0 500 780 full]
"""
def convertColorSpace(I,space = 'HLS', cvload = True):
    if space == 'RGB':
        if cvload:
            ret = cv2.cvtColor(I,cv2.COLOR_BGR2RGB)
        else:
            ret = I
    elif space == 'gray':
        ret = cv2.cvtColor(cv2.cvtColor(I,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
        # add color channels:
    elif space == 'HLS':
        ret = cv2.cvtColor(I,cv2.COLOR_BGR2HLS)
    elif space == 'HSV':
        ret = cv2.cvtColor(I,cv2.COLOR_BGR2HSV)
    elif space == 'LUV':
        ret = cv2.cvtColor(I,cv2.COLOR_BGR2LUV)
    elif space == 'YUV':
        ret = cv2.cvtColor(I,cv2.COLOR_BGR2YUV)
    elif space == 'YCrCb':
        ret = cv2.cvtColor(I,cv2.COLOR_BGR2YCrCb)
    return ret

def getMask(I):
    mask = np.zeros(I.shape, dtype=np.uint8)
    channel_count = I.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    roi_corners = np.array([[(0,720), (500,345), (780,345),(1280,720)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    return mask

def sobel(img, dir = 'x'):
     # Sobel x
     if dir == 'x':
         sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0) # Take the derivative in x
     else:
         sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1) # Take the derivative in y
     abs_sobel = np.absolute(sobel) # Absolute x derivative to accentuate lines away from horizontal
     scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
     return scaled_sobel

def process(I, mask = None):
    
    sobs = []
    sobs.append(sobel(I)) # BGR
    sobs.append(sobel(convertColorSpace(I,'gray')))
#    lap = cv2.Laplacian(convertColorSpace(I,'gray'),cv2.CV_64F)
#    sobs.append(sobel(convertColorSpace(I,'HSV')))
#    sobs.append(sobel(convertColorSpace(I,'LUV')))
#    sobs.append(sobel(convertColorSpace(I,'YUV')))
#    sobs.append(sobel(convertColorSpace(I,'YCrCb')))
    # mask:
    sobs[:] = [cv2.bitwise_and(s,mask) for s in sobs]

    ret = np.concatenate((sobs[0],sobs[1]),axis=0)
    ret = cv2.resize(ret,(ret.shape[0]/2,ret.shape[1]/2))
   
    
    return ret


mask = np.zeros(I.shape, dtype=np.uint8)
m = np.float32([
                [0, 720],
                [500, 345],
                [780, 345],
                [1280, 720]])
channel_count = I.shape[2]  # i.e. 3 or 4 depending on your image
ignore_mask_color = (255,)*channel_count
roi_corners = np.array([[(0,720), (500,345), (780,345),(1280,720)]], dtype=np.int32)
cv2.fillPoly(mask, roi_corners, ignore_mask_color)
# from Masterfool: use cv2.fillConvexPoly if you know it's convex
# apply the mask
masked_image = cv2.bitwise_and(I, mask)

# run processor on video stream and visualize, tune to find optimal 

def test():
    cap = cv2.VideoCapture('video_files/skytrain.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    ret, I = cap.read()
    maskTrack = getMask(I)
    while ret:
        ret, I = cap.read()
        P = process(I, maskTrack)
        cv2.imshow("processed", P)
        cv2.waitKey(5)
        


def spaceExplorer():
    # load each image, process through color transform, plot each channel output as a 1x3 figure:
    cspaces = ['RGB','HLS','HSV','YUV','LUV','YCrCb']
    files = glob.glob('*.png')
    
    i = 0
    for f in files:
        I = np.uint8(255*mpimg.imread(f)) # (may want to convert from (0,1) to (0,255))
        plt.close('all')
        fig, ax = plt.subplots(6, 3, figsize=(9, 24), num = 1)
        
        k = 0
        for c in cspaces:
            temp = convertColorSpace(I,c)
            ax[k,0].imshow(temp[:,:,0], cmap='gray')
            ax[k,0].axis('off')
            ax[k,1].imshow(temp[:,:,1], cmap='gray')
            ax[k,1].axis('off')
            ax[k,2].imshow(temp[:,:,2], cmap='gray')
            ax[k,2].axis('off')
            k += 1
        fig.tight_layout
        i += 1

"""
Process Idea

take gray image (or HLS?)
use Left/ right track separate seeds
for each lane;
    grab bottom 10% of pixles near seed location
    verticall sum them, take pixel pose with largest response
    use that pose and estimate track width as start pose for Y vertical pixels
    then scan up from first seed width up to vertical limit

May be better to find exactly which channel x-sobel should be applied on, what color range in color space

Rail Properties
HSV:
    low saturation < 90
HSL:
    low saturation < 50

CMYK:
    K value (110-172)
    M < 100
"""