#Text image interpreter
#By Tim Chinenov
#import Letter
#import statistics
import cv2
import numpy as np
from matplotlib import pyplot as plt

#function finds the corners given the top,bottom,left,and right
#maximum pixels
def findCorners(bound):
    c1 = [bound[3][0],bound[0][1]]
    c2 = [bound[1][0],bound[0][1]]
    c3 = [bound[1][0],bound[2][1]]
    c4 = [bound[3][0],bound[2][1]]
    return [c1,c2,c3,c4]

#function finds the minimization of the weighted within-class variance
#this algorithm is adapted from:
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
def findThresh(data):
    Binsize = 50
    #find density and bounds of histogram of data
    density,bds = np.histogram(data,bins=Binsize)
    #normalize the histogram values
    norm_dens = (density)/float(sum(density))
    #find discrete cumulative density function
    cum_dist = norm_dens.cumsum()
    #initial values to be overwritten
    fn_min = np.inf
    thresh = -1
    bounds = range(1,Binsize)
    #begin minimization routine
    for itr in range(0,Binsize):
        if(itr == Binsize-1):
            break;
        p1 = np.asarray(norm_dens[0:itr])
        p2 = np.asarray(norm_dens[itr+1:])
        q1 = cum_dist[itr]
        q2 = cum_dist[-1] - q1
        b1 = np.asarray(bounds[0:itr])
        b2 = np.asarray(bounds[itr:])
        #find means
        m1 = np.sum(p1*b1)/q1
        m2 = np.sum(p2*b2)/q2
        #find variance
        v1 = np.sum(((b1-m1)**2)*p1)/q1
        v2 = np.sum(((b2-m2)**2)*p2)/q2

        #calculate minimization function and replace values
        #if appropriate
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = itr

    return thresh,bds[thresh]

def dist(P1,P2):
    return np.sqrt((P1[0]-P2[0])**2+(P1[1]-P2[1])**2)

#function takes two rectangles of corners and combines them into a single
#rectangle
def mergeBoxes(c1,c2):
    newRect = []
    #find new corner for the top left
    cx = min(c1[0][0],c2[0][0])
    cy = min(c1[0][1],c2[0][1])
    newRect.append([cx,cy])
    #find new corner for the top right
    cx = max(c1[1][0],c2[1][0])
    cy = min(c1[1][1],c2[1][1])
    newRect.append([cx,cy])
    #find new corner for bottm right
    cx = max(c1[2][0],c2[2][0])
    cy = max(c1[2][1],c2[2][1])
    newRect.append([cx,cy])
    #find new corner for bottm left
    cx = min(c1[3][0],c2[3][0])
    cy = max(c1[3][1],c2[3][1])
    newRect.append([cx,cy])
    return newRect

#given a list of corners that represent the corners of a box,
#find the center of that box
def findCenterCoor(c1):
    width = abs(c1[0][0]-c1[1][0])
    height = abs(c1[0][1]-c1[3][1])
    return([c1[0][0]+(width/2.0), c1[0][1]+(height/2.0)])

#take two points and find their slope
def findSlope(p1,p2):
    if(p1[0]-p2[0] == 0):
        return np.inf

    return (p1[1]-p2[1])/(p1[0]-p2[0])

#takes point and set of corners and checks if the point is within the bounds
def isInside(p1,c1):
    if(p1[0] >= c1[0][0] and p1[0] <= c1[1][0] and p1[1] >= c1[0][1] and p1[1] <= c1[2][1]):
        return True
    else:
        return False

def findArea(c1):
    return abs(c1[0][0]-c1[1][0])*abs(c1[0][1]-c1[3][1])