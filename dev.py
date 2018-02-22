#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Development workspace
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob 

import imageio
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

# CURVE:
filename = 'skytrain.mp4'
clip = VideoFileClip(filename)
clip1 = clip.cutout(0, 135)
clip1 = clip1.cutout(35,clip1.duration)
clip1.write_videofile("test2.mp4")


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
channel_count = I.shape[2]  # i.e. 3 or 4 depending def convertColorSpace(I,space = 'HLS', cvload = True):
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

def generic_threshold(img, thresh = []):def convertColorSpace(I,space = 'HLS', cvload = True):
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
on your image
ignore_mask_color = (255,)*channel_count
roi_corners = np.array([[(0,720), (500,345), (780,345),(1280,720)]], dtype=np.int32)
cv2.fillPoly(mask, roi_corners, ignore_mask_color)
# from Masterfool: use cv2.fillConvexPoly if you know it's convex
# apply the mask
masked_image = cv2.bitwise_and(I, mask)

# run processor on video stream and visualize, tune to find optimal 

def test():
    cap = cv2.VideoCapture('video_files/test.mp4')
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
    cap = cv2.VideoCapture('skytrain.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    ret, I = cap.read()
    mask = getMaskTrack(I)
    maskTrack = getMask(I)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    while ret:
        ret, I = cap.read()
        Ic = I.copy();
        # take cropbox out:
        rect = I[550:720,256:1024,:]
        gray = cv2.cvtColor(rect,cv2.COLOR_RGB2HSV)
        out = gray[:,:,2]
        out = clahe.apply(out)
        out = cv2.blur(out,(5,5))
        # put rect back:
        rstack = np.dstack((out,out,out))
        I[550:720,256:1024,:] = rstack
        
        
        edges = cv2.Canny(I[:,:,1],10,150,apertureSize = 5)
        # mask edges:
        edges = cv2.bitwise_and(edges,mask[:,:,0])
        
        lines = (cv2.HoughLines(edges,1,np.pi/180,50))
        L = []
        if lines is not None:
            for l in range(0,lines.shape[0]):
                rho = lines[l,:,0]
                theta = lines[l,:,1]
                if abs(theta-np.pi/2) > 1:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    y1 = 720
                    y2 = 600
                    if abs(b)>1e-4:
                        m = a/-b
                        yint = y0-x0*m      
                        x1 = int((y1-yint)/m)
                        x2 = int((y2-yint)/m)
                    else:
                        x1 = x0
                        x2 = x0
                        
                    L.append([x1,y1,x2,y2])
                    cv2.line(I,(x1,y1),(x2,y2),(0,0,255),2)
#        Ig = cv2.cvtColor(Ic,cv2.COLOR_RGB2HSV)
#        Igc = clahe.apply(Ig[:,:,2])
##        Igc = cv2.blur(Igc,(5,5))
##        E2 = cv2.Canny(Igc,10,150,apertureSize = 5)
#        E2 = np.uint8(255*generic_threshold(sobel(Igc), (90,255)))
#        Iout = np.dstack([E2,E2,E2])
        cv2.imshow("processed", I)
        cv2.waitKey(5)
      
from imutils.object_detection import non_max_suppression
import imutils
def test3(): #person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    cap = cv2.VideoCapture('video_files/skytrain.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    ret, I = cap.read()
    while ret:
        ret, image = cap.read()
        image = imutils.resize(image, width=min(400, image.shape[1]))
        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
         
#        # draw the original bounding boxes
#        for (x, y, w, h) in rects:
#            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
         
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        cv2.imshow("processed", image)
        cv2.waitKey(5)


def test4(): # testing with patch matching / Convolution
    I = cv2.imread('test_output/frame_00.png')
    HSV = cv2.cvtColor(I,cv2.COLOR_BGR2HLS) 
    plt.figure(1)
    plt.clf
    plt.imshow(HSV[:,:,1], cmap='gray')
    
    
    p_L = HSV[670:720,395:450,1]
    p_R = HSV[670:720,730:785,1]
    (tH, tW) = p_L.shape[:2]
    rail_width = 350
#    plt.figure(4)
#    plt.clf
#    plt.imshow(cv2.blur(p_R,(3,3)), cmap='gray')
    
    I_bot = HSV[570:620,:,1]
    # LEFT
    result_L = cv2.matchTemplate(I_bot, p_L, cv2.TM_CCORR_NORMED)
    (_, maxVal_L, _, maxLoc_L) = cv2.minMaxLoc(result_L)
    # RIGHT
    result_R = cv2.matchTemplate(I_bot, p_R, cv2.TM_CCORR_NORMED)
    (_, maxVal_R, _, maxLoc_R) = cv2.minMaxLoc(result_R)
    
    # take the normalized sum of the results with offset of 350 pixels to find most likely pose of both rails:
    
    rL = (result_L[0,:]-result_L[0,:].min())/(result_L[0,:]-result_L[0,:].min()).max()
    rR = (result_R[0,:]-result_R[0,:].min())/(result_R[0,:]-result_R[0,:].min()).max()
    #sum of L/R responses with offset with rail_width:
    # technically, the max of these will set initial position, but may need to sweep rail_width...
    rsums = rL[:-rail_width] + rR[rail_width:]
    
    #sweeping rail width:
    wwopt = [(rL[:-ww] + rR[ww:]).max() for ww in range(350-20,350+20)]
    # max:
    pos = [i for i in range(0,len(wwopt)) if wwopt[i]==max(wwopt)]
    
    www = 350-20 + pos[0]
    rsums = rL[:-www] + rR[www:]
    
    
    plt.figure(2)
    plt.clf()
    plt.plot(np.linspace(1,len(rL),len(rL)),rL,'*b')
    plt.plot(np.linspace(1,len(rR),len(rR)),rR,'*r')
    plt.plot(np.linspace(1,len(rsums),len(rsums)),rsums,'*k')
    
    
    
#    clone = np.dstack([I_bot, I_bot, I_bot])
#    cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
#    plt.figure(3)
#    plt.clf
#    plt.imshow(clone, cmap='gray')
    
    # Hard set L/R:
    maxLoc_L = [395,0]
    maxLoc_R = [730,0]
    # run on video:
    cap = cv2.VideoCapture('skytrain.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES,20)
    ret, I = cap.read()
    while ret:
        ret, I = cap.read()
        HLS = cv2.cvtColor(I,cv2.COLOR_BGR2HLS)
        # left side:
        x_min_L = maxLoc_L[0] - 10
        x_max_L = maxLoc_L[0] + tW + 10
        I_bot_L = HLS[670:720,x_min_L:x_max_L,1]
        result_L = cv2.matchTemplate(I_bot_L, p_L, cv2.TM_CCORR_NORMED)
        (_, maxVal_L, _, maxLoc_L) = cv2.minMaxLoc(result_L)
        cv2.rectangle(I, (x_min_L + maxLoc_L[0], 670+maxLoc_L[1]),(x_min_L + maxLoc_L[0] + tW, 670+maxLoc_L[1] + tH), (0, 0, 255), 2)
        maxLoc_L = [maxLoc_L[0] + x_min_L, maxLoc_L[1]]
        # right side:
        x_min_R = maxLoc_R[0] - 10
        x_max_R = maxLoc_R[0] + tW + 10
        I_bot_R = HLS[670:720,x_min_R:x_max_R,1]
        result_R = cv2.matchTemplate(I_bot_R, p_R, cv2.TM_CCORR_NORMED)
        (_, maxVal_R, _, maxLoc_R) = cv2.minMaxLoc(result_R)
        cv2.rectangle(I, (x_min_R + maxLoc_R[0], 670+maxLoc_R[1]),(x_min_R + maxLoc_R[0] + tW, 670+maxLoc_R[1] + tH), (0, 255, 0), 2)
        maxLoc_R = [maxLoc_R[0] + x_min_R, maxLoc_R[1]]
        # add more layers:
        x_min_L = maxLoc_L[0]  + 20
        x_max_L = maxLoc_L[0] + tW + 30
        I_bot_L = HLS[670-tH:720-tH,x_min_L:x_max_L,1]
        result_L = cv2.matchTemplate(I_bot_L, p_L, cv2.TM_CCORR_NORMED)
        (_, maxVal_L, _, maxLoc_L2) = cv2.minMaxLoc(result_L)
        cv2.rectangle(I, (x_min_L + maxLoc_L2[0], 670+maxLoc_L2[1]-tH),(x_min_L + maxLoc_L2[0] + tW, 670+maxLoc_L2[1]), (0, 0, 255), 2)
#        maxLoc_L = [maxLoc_L[0] + x_min_L, maxLoc_L[1]]
        # right side:
        x_min_R = maxLoc_R[0] - 30
        x_max_R = maxLoc_R[0] + tW -20 
        I_bot_R = HLS[670-tH:720-tH,x_min_R:x_max_R,1]
        result_R2 = cv2.matchTemplate(I_bot_R, p_R, cv2.TM_CCORR_NORMED)
        (_, maxVal_R, _, maxLoc_R2) = cv2.minMaxLoc(result_R2)
        cv2.rectangle(I, (x_min_R + maxLoc_R2[0], 670+maxLoc_R2[1]-tH),(x_min_R + maxLoc_R2[0] + tW, 670+maxLoc_R2[1]), (0, 255, 0), 2)
#        maxLoc_R = [maxLoc_R[0] + x_min_R, maxLoc_R[1]]
            
        # Display
        cv2.imshow("processed", I)
        cv2.waitKey(5)

def detectOnLine(image = None , template = None ,x_start = 0, x_width = 10, y = 0): 
    """
    Opencv Template-Matching detector, y_range == template.height
    x_start is start location to search, x_width is the padding around tamplate to use
    x_start must be >= 0
    Patch is take from y and downward!
    Pass full image in...
    """
    raise NameError('No input image') if image is None else pass
    raise NameError('No template image') if template is None else pass
    (h,w) = image.shape[:2]
    (tH, tW) = template.shape[:2]
    # throw error is y+tH > h!
    raise NameError('y + tH > h!') if y+tH>h else pass
    
    x_min = x_start if x_start-x_width < 0 else x_start-x_width
    x_max = w is x_start+x_witdh+tW > w else x_start+x_witdh+tW 
    # take a portion of the image that is the height of the patch
    imagePatch = image[y:y+tH,x_min:x_max]
    # Use opencv Template matching normalized cross-correlation:
    result = cv2.matchTemplate(imagePatch, template, cv2.TM_CCORR_NORMED)
    # find the best response:
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    # Get bounding rectangle for match in IMAGE COORDINATES:
    rect = [x_min + maxLoc[0], y+maxLoc[1], x_min + maxLoc[0] + tW, y+maxLoc[1] + tH]
    maxLoc = [maxLoc[0] + x_min_L, maxLoc[1]]
    return maxLoc, rect

def getTemplate(image = None, ch = 0, y_range = [0,0], x_range = [0,0], space = 'HLS', ocv = False):
    raise NameError('No input image') if image is None else pass
    I = convertColorSpace(image,space = space, cvload = ocv)
    # raise some logical errors:
    raise NameError('y range < 0!') if y_range[1]-y_range[0]<=0 else pass
    raise NameError('x range < 0!') if x_range[1]-x_range[0]<=0 else pass

    raise NameError('y range is out of image!') if (y_range[0]<=0 or y_range[1]>image.shape[0]) else pass
    raise NameError('x range is out of image!') if (x_range[0]<=0 or x_range[1]>image.shape[1]) else pass

    patch = I[y_range[0]:y_range[1],x_range[0]:x_range[1],ch]
    
    return patch
    


class rail():
    def __init__(self, template = None):
        self.template = None
        self.pose = [0,0]

class railDetector():
    def __init__(self, L = 395, R = 710, w = 350, usingOCV = False):
        # may want to automate this part to grab initial templates:
        I = cv2.imread('test_output/frame_00.png')
        HLS = cv2.cvtColor(I,cv2.COLOR_BGR2HLS) 
        self.p_L = HLS[670:720,395:450,1]
        self.p_R = HLS[670:720,730:785,1]
        (self.tH, self.tW) = self.p_L.shape[:2]
        # Hard set L/R:
        self.R_init = R
        self.L_init = L
        self.maxLoc_L = [L,0]
        self.maxLoc_R = [R,0]
        self.w = w
        self.usingOCV = usingOCV
    

    
    def process(self,I):
        # color conversion:
        HLS = cv2.cvtColor(I,cv2.COLOR_BGR2HLS)
        """
        first layer: (may need an initialization with no seed available)
            if seed or previous available:
                get Right Patch Response by previous (or init) seed
                get Left Patch Response by previous (or init) seed
                using pre-set rail width in pixels, find responses that best fit width (max for both)
                (tracking with previous data) smooth with any previous data
            if no seed or previous available:
                TBD
        for each next layer:
            using first layer seed + search bnd:
                get Right Patch Response by previous (or init) seed
                get Left Patch Response by previous (or init) seed
                find best response with less than or equal to width from previous layer
        output is left, right tracks represented by matched patches
        next up, need to fit polygon to each lane:
            TBD! but general idea is to mask everything outside of matched blocks,
            find edges, use edge pixels to fit 2nd order poly along y-direction
        """
        # left side:
        x_min_L = self.maxLoc_L[0] - 10
        x_max_L = self.maxLoc_L[0] + tW + 10
        I_bot_L = HLS[670:720,x_min_L:x_max_L,1]
        result_L = cv2.matchTemplate(I_bot_L, self.p_L, cv2.TM_CCORR_NORMED)
        (_, maxVal_L, _, maxLoc_L) = cv2.minMaxLoc(result_L)
        cv2.rectangle(I, (x_min_L + maxLoc_L[0], 670+maxLoc_L[1]),(x_min_L + maxLoc_L[0] + tW, 670+maxLoc_L[1] + tH), (0, 0, 255), 2)
        self.maxLoc_L = [maxLoc_L[0] + x_min_L, maxLoc_L[1]]
        # right side:
        x_min_R = self.maxLoc_R[0] - 10
        x_max_R = self.maxLoc_R[0] + tW + 10
        I_bot_R = HLS[670:720,x_min_R:x_max_R,1]
        result_R = cv2.matchTemplate(I_bot_R, self.p_R, cv2.TM_CCORR_NORMED)
        (_, maxVal_R, _, maxLoc_R) = cv2.minMaxLoc(result_R)
        cv2.rectangle(I, (x_min_R + maxLoc_R[0], 670+maxLoc_R[1]),(x_min_R + maxLoc_R[0] + tW, 670+maxLoc_R[1] + tH), (0, 255, 0), 2)
        self.maxLoc_R = [maxLoc_R[0] + x_min_R, maxLoc_R[1]]
        # add more layers:
        x_min_L = self.maxLoc_L[0]  + 20
        x_max_L = self.maxLoc_L[0] + tW + 30
        I_bot_L = HLS[670-tH:720-tH,x_min_L:x_max_L,1]
        result_L = cv2.matchTemplate(I_bot_L, self.p_L, cv2.TM_CCORR_NORMED)
        (_, maxVal_L, _, maxLoc_L2) = cv2.minMaxLoc(result_L)
        cv2.rectangle(I, (x_min_L + maxLoc_L2[0], 670+maxLoc_L2[1]-tH),(x_min_L + maxLoc_L2[0] + tW, 670+maxLoc_L2[1]), (0, 0, 255), 2)
        maxLoc_L2 = [maxLoc_L2[0] + x_min_L, maxLoc_L2[1]]
        # right side:
        x_min_R = self.maxLoc_R[0] - 30
        x_max_R = self.maxLoc_R[0] + tW -20 
        I_bot_R = HLS[670-tH:720-tH,x_min_R:x_max_R,1]
        result_R2 = cv2.matchTemplate(I_bot_R, self.p_R, cv2.TM_CCORR_NORMED)
        (_, maxVal_R, _, maxLoc_R2) = cv2.minMaxLoc(result_R2)
        cv2.rectangle(I, (x_min_R + maxLoc_R2[0], 670+maxLoc_R2[1]-tH),(x_min_R + maxLoc_R2[0] + tW, 670+maxLoc_R2[1]), (0, 255, 0), 2)
        maxLoc_R2 = [maxLoc_R2[0] + x_min_R, maxLoc_R2[1]]
        
        x_min_L = maxLoc_L2[0]  + 15
        x_max_L = maxLoc_L2[0] + tW + 25
        I_bot_L = HLS[670-2*tH:720-2*tH,x_min_L:x_max_L,1]
        result_L = cv2.matchTemplate(I_bot_L, self.p_L, cv2.TM_CCORR_NORMED)
        (_, maxVal_L, _, maxLoc_L2) = cv2.minMaxLoc(result_L)
        cv2.rectangle(I, (x_min_L + maxLoc_L2[0], 670+maxLoc_L2[1]-2*tH),(x_min_L + maxLoc_L2[0] + tW, 670+maxLoc_L2[1]-tH), (0, 0, 255), 2)
        maxLoc_L2 = [maxLoc_L2[0] + x_min_L, maxLoc_L2[1]]
#        # right side:
        x_min_R = maxLoc_R2[0] - 25
        x_max_R = maxLoc_R2[0] + tW -15 
        I_bot_R = HLS[670-2*tH:720-2*tH,x_min_R:x_max_R,1]
        result_R2 = cv2.matchTemplate(I_bot_R, self.p_R, cv2.TM_CCORR_NORMED)
        (_, maxVal_R, _, maxLoc_R2) = cv2.minMaxLoc(result_R2)
        cv2.rectangle(I, (x_min_R + maxLoc_R2[0], 670+maxLoc_R2[1]-2*tH),(x_min_R + maxLoc_R2[0] + tW, 670+maxLoc_R2[1]-tH), (0, 255, 0), 2)
        maxLoc_R2 = [maxLoc_R2[0] + x_min_R, maxLoc_R2[1]]
        
        x_min_L = maxLoc_L2[0]  + 15
        x_max_L = maxLoc_L2[0] + tW + 25
        I_bot_L = HLS[670-3*tH:720-3*tH,x_min_L:x_max_L,1]
        result_L = cv2.matchTemplate(I_bot_L, self.p_L, cv2.TM_CCORR_NORMED)
        (_, maxVal_L, _, maxLoc_L2) = cv2.minMaxLoc(result_L)
        cv2.rectangle(I, (x_min_L + maxLoc_L2[0], 670+maxLoc_L2[1]-3*tH),(x_min_L + maxLoc_L2[0] + tW, 670+maxLoc_L2[1]-2*tH), (0, 0, 255), 2)
        maxLoc_L2 = [maxLoc_L2[0] + x_min_L, maxLoc_L2[1]]
#        # right side:
        x_min_R = maxLoc_R2[0] - 25
        x_max_R = maxLoc_R2[0] + tW -15 
        I_bot_R = HLS[670-3*tH:720-3*tH,x_min_R:x_max_R,1]
        result_R2 = cv2.matchTemplate(I_bot_R, self.p_R, cv2.TM_CCORR_NORMED)
        (_, maxVal_R, _, maxLoc_R2) = cv2.minMaxLoc(result_R2)
        cv2.rectangle(I, (x_min_R + maxLoc_R2[0], 670+maxLoc_R2[1]-3*tH),(x_min_R + maxLoc_R2[0] + tW, 670+maxLoc_R2[1]-2*tH), (0, 255, 0), 2)
        maxLoc_R2 = [maxLoc_R2[0] + x_min_R, maxLoc_R2[1]]
        
        return I



detector = railDetector(R = 720)
cap = cv2.VideoCapture('skytrain.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES,20)
ret, I = cap.read()
while ret:
    ret, I = cap.read()
    cv2.imshow("processed", detector.process(I))
    cv2.waitKey(5)

# process:
fileName = 'test2.mp4'
write_output = 'test_output/' + fileName
clip1 = VideoFileClip(fileName)
firstImage = clip1.get_frame(0)
# make rail tracker object:
detector = railDetector()
white_clip = clip1.fl_image(detector.process)
white_clip.write_videofile(write_output, audio=False)

