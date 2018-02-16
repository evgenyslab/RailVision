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

def getMaskTrack(I):
    mask = np.zeros(I.shape, dtype=np.uint8)
    channel_count = I.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    roi_corners = np.array([[(360,720), (360,620), (920,620),(920,720)]], dtype=np.int32)
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

def generic_threshold(img, thresh = []):
    """
    generic threshold method.
    Applies each threshold pair on image independently and returns images list with each threshold applied.
    """
    
    if type(thresh) is list:
        binary = []
        for t in thresh:
            temp = np.zeros_like(img)
            temp[(img >= t[0]) & (img <= t[1])] = 1
            binary.append(temp)
    else:
        binary = np.zeros_like(img)
        binary[(img >= thresh[0]) & (img <= thresh[1])] = 1
    
    return binary

def process(I, mask = None):
    
#    sobs = []
#    sobs.append(sobel(I)) # BGR
#    sobs.append(sobel(convertColorSpace(I,'gray')))
##    lap = cv2.Laplacian(convertColorSpace(I,'gray'),cv2.CV_64F)
##    sobs.append(sobel(convertColorSpace(I,'HSV')))
##    sobs.append(sobel(convertColorSpace(I,'LUV')))
##    sobs.append(sobel(convertColorSpace(I,'YUV')))
##    sobs.append(sobel(convertColorSpace(I,'YCrCb')))
#    # mask:
#    sobs[:] = [cv2.bitwise_and(s,mask) for s in sobs]

    HSV = convertColorSpace(I,'HSV')
    # threshold 2 channels:
    ch1 = generic_threshold(HSV[:,:,1], (0,90))
    ch2 = generic_threshold(HSV[:,:,2], (90,120))
    
    P = np.bitwise_xor(ch1,ch2);
    
    P = np.uint8(np.dstack([P,P,P])*255)

    ret = np.concatenate((I,P),axis=0)
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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    while ret:
        ret, I = cap.read()
        # take cropbox out:
        rect = I[550:720,256:1024,:]
        gray = cv2.cvtColor(rect,cv2.COLOR_RGB2HSV)
        out = gray[:,:,2]
        
        out = clahe.apply(out)
        out = cv2.blur(out,(5,5))
        # put rect back:
        rstack = np.dstack((out,out,out))
        I[550:720,256:1024,:] = rstack
#        out *= 255/out.max()
        cv2.imshow("processed", I)
        cv2.waitKey(5)
        


def spaceExplorer():
    # load each image, process through color transform, plot each channel output as a 1x3 figure:
    cspaces = ['RGB','HLS','HSV','YUV','LUV']
    files = glob.glob('*.png')
    
    i = 0
    for f in files:
        I = np.uint8(255*mpimg.imread(f)) # (may want to convert from (0,1) to (0,255))
        plt.close('all')
        fig, ax = plt.subplots(5, 3, figsize=(9, 24), num = 1,sharex=True, sharey=True)
        
        k = 0
        for c in cspaces:
            temp = convertColorSpace(I,c)
            for j in range(0,3):
                p = sobel(temp[:,:,j])
                p *= 255/p.max()
#                p = generic_threshold(p,thresh = (0,255))
#                p = temp[:,:,j]
                ax[k,j].imshow(p, cmap='gray')
                # ax[k,j].axis('off')
        
            k += 1
        i += 1



I = np.uint8(255*mpimg.imread('frame_08.png'))
mask = getMaskTrack(I)
gray = cv2.cvtColor(I,cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(I,10,150,apertureSize = 3)
# mask edges:
edges = cv2.bitwise_and(edges,mask[:,:,0])
lines = (cv2.HoughLines(edges,1,np.pi/180,100)).squeeze()
for l in lines:
    rho = l[0]
    theta = l[1]
    if abs(theta-np.pi) > 0.2:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        m = a/-b
        yint = y0-x0*m
        
        y1 = 720
        x1 = int((y1-yint)/m)
        y2 = 350
        x2 = int((y2-yint)/m)
   
        cv2.line(I,(x1,y1),(x2,y2),(0,0,255),2)

plt.close('all')
plt.figure(1)
plt.imshow(edges, cmap='gray')
plt.figure(2)
plt.imshow(I)


def test2():
    cap = cv2.VideoCapture('video_files/skytrain.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    ret, I = cap.read()
    mask = getMaskTrack(I)
    while ret:
        ret, I = cap.read()
        gray = cv2.cvtColor(I,cv2.COLOR_RGB2HSV)
        edges = cv2.Canny(gray[:,:,1],10,150,apertureSize = 3)
        # mask edges:
        edges = cv2.bitwise_and(edges,mask[:,:,0])
        lines = (cv2.HoughLines(edges,1,np.pi/180,100))
        if lines is not None:
            for l in range(0,lines.shape[0]):
                rho = lines[l,:,0]
                theta = lines[l,:,1]
                if abs(theta-np.pi/2) > 0.2:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    y1 = 720
                    y2 = 500
                    if abs(b)>1e-4:
                        m = a/-b
                        yint = y0-x0*m      
                        x1 = int((y1-yint)/m)
                        x2 = int((y2-yint)/m)
                    else:
                        x1 = x0
                        x2 = x0
                        
               
                    cv2.line(I,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow("processed", I)
        cv2.waitKey(5)


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
    
    S < 90  + 90 < v < 120
    
HSL:
    low saturation < 50

CMYK:
    K value (110-172)
    M < 100
"""