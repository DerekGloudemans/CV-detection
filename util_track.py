from __future__ import division
from itertools import combinations
from scipy.optimize import linear_sum_assignment
import time
import torch
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import random


def condense_detections(detections,style = "center"):
    """
    converts the input data object (from the yolo detector or similar) into the 
    specified output style
    
    detections - input list of length equal to number of frames. Each list item is
    a D x 8 numpy array with a row for each object containing:
    index of the image in the batch (always 0 in this implementation 
    4 corner coordinates (x1,y1,x2,y2), objectness score, the score of class 
    with maximum confidence, and the index of that class.
    
    will return a list of D x ? numpy arrays with the contents of each row as 
    specified by style parameter
    
    style "center" -  centroid x, centroid y
    style "bottom_center" - centroid x, bottom y
    style "SORT" - centroid x, centroid y, scale (height) and ratio (width/height)
    style "SORT_with_conf" - as above plus detection confidence
    """
    assert style in ["SORT_with_conf","SORT","center","bottom_center"], "Invalid style input."
    
    new_list = []
    
    if style == "center":
        for item in detections:
            coords = np.zeros([len(item),2])
            for i in range(0,len(item)):
                coords[i,0] = (item[i,1]+item[i,3])/2.0
                coords[i,1] = (item[i,2]+item[i,4])/2.0
            new_list.append(coords) 
            
    elif style == "bottom_center":   
        for item in detections:
            coords = np.zeros([len(item),2])
            for i in range(0,len(item)):
                coords[i,0] = (item[i,1]+item[i,3])/2.0
                coords[i,1] = item[i,4]
            new_list.append(coords)
            
    elif style == "SORT":
        for item in detections:
            coords = np.zeros([len(item),4])
            for i in range(0,len(item)):
                coords[i,0] = (item[i,1]+item[i,3])/2.0 # x centroid
                coords[i,1] = (item[i,2]+item[i,4])/2.0 # y centroid
                coords[i,2] = (item[i,4]-item[i,2]) # scale (y height)
                coords[i,3] = (item[i,3]-item[i,1])/float(coords[i,2]) # ratio (width/height)
            new_list.append(coords)
            
    elif style == "SORT_with_conf":
        for item in detections:
            coords = np.zeros([len(item),4])
            for i in range(0,len(item)):
                coords[i,0] = (item[i,1]+item[i,3])/2.0 # x centroid
                coords[i,1] = (item[i,2]+item[i,4])/2.0 # y centroid
                coords[i,2] = (item[i,4]-item[i,2]) # scale (y height)
                coords[i,3] = (item[i,3]-item[i,1])/float(coords[i,2]) # ratio (width/height)
                coords[i,4] = (item[i,5])
            new_list.append(coords)
       
    return new_list


def match_greedy(first,second,threshold = 10):
    """
    performs  greedy best-first matching of objects between frames
    inputs - N x 2 arrays of object x and y coordinates from different frames
    output - M x 1 array where index i corresponds to the second frame object 
    matched to the first frame object i
    """
    


    
    # find distances between first and second
    dist = np.zeros([len(first),len(second)])
    for i in range(0,len(first)):
        for j in range(0,len(second)):
            dist[i,j] = np.sqrt((first[i,0]-second[j,0])**2 + (first[i,1]-second[j,1])**2)
    
    # select closest pair
    matchings = np.zeros(len(first))-1
    unflat = lambda x: (x//len(second), x %len(second))
    while np.min(dist) < threshold:
        min_f, min_s = unflat(np.argmin(dist))
        #print(min_f,min_s,len(first),len(second),len(matchings),np.argmin(dist))
        matchings[min_f] = min_s
        dist[:,min_s] = np.inf
        dist[min_f,:] = np.inf
        
    return matchings


def match_hungarian(first,second):
    """
    performs  optimal (in terms of sum distance) matching of points 
    in first to second using the Hungarian algorithm
    inputs - N x 2 arrays of object x and y coordinates from different frames
    output - M x 1 array where index i corresponds to the second frame object 
    matched to the first frame object i
    """
    # find distances between first and second
    dist = np.zeros([len(first),len(second)])
    for i in range(0,len(first)):
        for j in range(0,len(second)):
            dist[i,j] = np.sqrt((first[i,0]-second[j,0])**2 + (first[i,1]-second[j,1])**2)
            
    _, matchings = linear_sum_assignment(dist)
    return matchings


def match_all(coords_list,match_fn = match_greedy):
    """
    performs matching using the match_fn strategy for all pairs of consecutive
    coordinate sets in coords_list
    coords_list- list of M x (x,y) pairs
    output - list of matchings between frames
    """
    out_list = []
    for i in range(0,len(coords_list)-1):
        
        first = coords_list[i]
        second = coords_list[i+1]
        out_list.append(match_fn(first,second))
    return out_list


def get_objects(matchings, coords,snap_threshold = 30, frames_lost_lim = 20):
    """
    Uses matchings to find all unique objects accross multiple frames
    """
    active_objs = []
    inactive_objs = []
    
    # initialize with all objects found in first frame
    for i,row in enumerate(coords[0]):
        obj = {
                'current': (row[0],row[1]),
                'all': [], # keeps track of all positions of the object
                'obj_id': i, # position in coords list
                'fsld': 0,   # frames since last detected
                'first_frame': 0 # frame in which object is first detected
                }
        obj['all'].append(obj['current']) 
        active_objs.append(obj)
    
    # for one set of matchings between frames - this loop will run (# frames -1) times
    # f is the frame number
    # frame_set is the 1 x objects numopy array of matchings from frame f to frame f+1
    for f, frame_set in enumerate(matchings):
        move_to_inactive = []
        matched_in_next = [] # keeps track of which objects from next frame are already dealt with
        
        # first, deal with all existing objects (each loop deals with one object)
        # o is the objecobjs[4]t index
        # obj is the object 
        for o, obj in enumerate(active_objs):
            matched = False
            obj_id_next = int(frame_set[obj['obj_id']]) # get the index of object in next frame or -1 if not matched in next frame
            # first deals with problem of reverse indexing, second verifies there is a match
            if obj['obj_id'] != -1 and obj_id_next != -1: # object has a match
                obj['obj_id'] = obj_id_next
                obj['current'] = (coords[f+1][obj_id_next,0],coords[f+1][obj_id_next,1])
                obj['all'].append(obj['current'])
                obj['fsld'] = 0
                
                matched_in_next.append(obj_id_next)
                matched = True
                
            else: # object does not have a certain match in next frame
                # search for match among detatched objects
                for j, row in enumerate(coords[f+1]):
                    if j not in matched_in_next: # not already matched to object in previous frame
                        # calculate distance to current obj
                        distance = np.sqrt((obj['current'][0]-row[0])**2 + (obj['current'][1]-row[1])**2)
                        if distance < snap_threshold: # close enough to call a match
                            obj['obj_id'] = j
                            obj['current'] = (row[0],row[1])
                            obj['all'].append(obj['current'])
                            obj['fsld'] = 0
                            
                            matched_in_next.append(j)
                            matched = True
                            break
                # if no match at all
                if not matched:
                    obj['obj_id'] = -1
                    obj['all'].append(obj['current']) # don't update location at all
                    obj['fsld'] += 1
                
                    if obj['fsld'] > frames_lost_lim:
                        move_to_inactive.append(o)
            
        # now, deal with objects found only in second frame  - each row is one object  
        for k, row in enumerate(coords[f+1]):
            # if row was matched in previous frame, the object has already been dealt with
            if k not in matched_in_next:
                # create a new object
                new_obj = {
                        'current': (row[0],row[1]),
                        'all': [], # keeps track of all positions of the object
                        'obj_id': k, # position in coords list
                        'fsld': 0,   # frames since last detected
                        'first_frame': f # frame in which object is first detected
                        }
                new_obj['all'].append(new_obj['current']) 
                active_objs.append(new_obj)
                
        # lastly, move all objects in move_to_inactive
        move_to_inactive.sort()
        move_to_inactive.reverse()
        for idx in move_to_inactive:
            inactive_objs.append(active_objs[idx])
            del active_objs[idx]
            
    return active_objs + inactive_objs


def extract_obj_coords(detections,pt_location = "center"):
    """ 
    wrapper function that condenses, matches, and extracts objects from a set
    of detections. pt_location = "center" or "bottom_center" and specifies
    where the object point should be placed
    returns point_array - a t x 2N array where t is the number of frames and N
    is the total number of unique objects detected in the video
    """
    coords = condense_detections(detections,pt_location)
    matchings = match_all(coords)   
    objs = get_objects(matchings, coords)

    # create an array where each row represents a frame and each two columns represent an object
    points_array = np.zeros([len(coords),len(objs)*2])-1
    for j in range(0,len(objs)):
        obj = objs[j]
        first_frame = int(obj['first_frame'])
        for i in range(0,len(obj['all'])):
            points_array[i+first_frame,j*2] = obj['all'][i][0]
            points_array[i+first_frame,(j*2)+1] = obj['all'][i][1]\
            
    return points_array, objs


# Kalman filter validation code
detections = np.load("temp_detections.npy",allow_pickle= True)
flattened = condense_detections(detections,style = "SORT")