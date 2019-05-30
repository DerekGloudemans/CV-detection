from __future__ import division
from itertools import combinations
import time
import torch
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import random

def detect_video(video_file, detector, verbose = True, show = True, save_file = None):
    
    # open up a videocapture object
    cap = cv2.VideoCapture(video_file)
    # verify file is opened
    assert cap.isOpened(), "Cannot open file \"{}\"".format(video_file)
    
    # opens VideoWriter object for saving video file if necessary
    if save_file != None:
        # open video_writer object
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(save_file,cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc('H','2','6','4'), 30, (frame_width,frame_height))
    
    #main loop   
    start = time.time()
    frames = 0
    ret = True
    all_detections = []
    
    # get first frame
    ret, frame = cap.read()
    
    while cap.isOpened():
        
        if ret:
            
            # detect frame
            detections,im_out = detector.detect(frame, show = False, verbose = False)
            if True: # convert to numpy array
                all_detections.append(detections.cpu().numpy())
            
            #summary statistics
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            
            # get next frame or None
            ret, frame = cap.read()
            
             # save frame to file if necessary
            if save_file != None:
                out.write(im_out)
            
            # output frame if necessary
            if show:
                im = cv2.resize(im_out, (1920, 1080))               
                cv2.imshow("frame", im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            
    # close all resources used      
    cap.release()
    cv2.destroyAllWindows()
    try:
        out.release()
    except:
        pass
    torch.cuda.empty_cache()
 
    print("Detection finished")
    return all_detections

def condense_detections(detections):
    """
    input - list of Dx8 numpy arrays correspoding to detections
    idx (always 0 in this imp.), 4 corner coordinates, objectness , score of class with max conf,class idx.
    output - list of D x 2 numpy arrays with x,y center coordinates
    """
    new_list = []
    for item in detections:
        coords = np.zeros([len(item),2])
        for i in range(0,len(item)):
            coords[i,0] = (item[i,1]+item[i,3])/2.0
            coords[i,1] = (item[i,2]+item[i,4])/2.0
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

def extract_obj_coords(detections):
    coords = condense_detections(detections)
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

def draw_track(point_array, file_in, file_out = None, show = False): 
    # load video file 
    cap = cv2.VideoCapture(file_in)
    assert cap.isOpened(), "Cannot open file \"{}\"".format(file_in)
    
    # opens VideoWriter object for saving video file if necessary
    if file_out != None:
        # open video_writer object
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(file_out,cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc('H','2','6','4'), 30, (frame_width,frame_height))
    
    # define random colors for each object
    colormaps = [(random.randrange(0,255),random.randrange(0,255), random.randrange(0,255)) for k in range(0,int(len(point_array[0])/2))]
    
    ret = True
    start = time.time()
    frame_num = 0
    
    # read first frame
    ret, frame = cap.read()
    
    while cap.isOpened():
        
        if ret:
            
            # Plot circles
            for i in range(0, int(len(point_array[0])/2)):
                try:
                    center = (int(point_array[frame_num,i*2]),int(point_array[frame_num,(i*2)+1]))
                    
                    cv2.circle(frame,center, 10, colormaps[i], thickness = -1)
                except:
                    pass # last frame is perhaps not done correctly
            im_out = frame #write here
            
            #summary statistics
            frame_num = frame_num + 1
            print("FPS of the video is {:5.2f}".format( frame_num / (time.time() - start)))
            
            # save frame to file if necessary
            if file_out != None:
                out.write(im_out)
            
            # get next frame or None
            ret, frame = cap.read()
            
            # output frame
            if show:
                im = cv2.resize(im_out, (1920, 1080))               
                cv2.imshow("frame", im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
                
        else:
            break
        
    # close all resources used      
    cap.release()
    cv2.destroyAllWindows()
    try:
        out.release()
    except:
        pass
    
    print("Tracking in camera-space finished.")
    
    
def draw_world(point_array, file_in, file_out = None, show = True):
    """
    outputs a video with points drawn on an image of the the world at each frame's 
    timestep
    """
    
    # load background image 
    world_im = cv2.imread(file_in)
    
    # opens VideoWriter object for saving video file if necessary
    if file_out != None:
        # open video_writer object
        frame_width = int(world_im.shape[1])
        frame_height = int(world_im.shape[0])
        out = cv2.VideoWriter(file_out,cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc('H','2','6','4'), 30, (frame_width,frame_height))
    
    # define random colors for each object
    colormaps = [(random.randrange(0,255),random.randrange(0,255), random.randrange(0,255)) for k in range(0,int(len(point_array[0])/2))]
    
    start = time.time()
    frame_num = 0
    # each loop processes one frame
    for fr in point_array:
            # create fresh copy of background
            frame = world_im.copy()
            
            # loop through points_array and plot circles on background
            for i in range(0, int(len(point_array[0])/2)):
                try:
                    center = (int(point_array[frame_num,i*2]),int(point_array[frame_num,(i*2)+1]))
                    cv2.circle(frame,center, 10, colormaps[i], thickness = -1)
                except:
                    pass # last frame is perhaps not done correctly, may also catch points that fall off image boundary
            
            #summary statistics
            frame_num += 1
            print("FPS of the video is {:5.2f}".format( frame_num / (time.time() - start)))
            
             # save frame to file if necessary
            if file_out != None:
                out.write(frame)
            
            # output frame
            if show:
                #im = cv2.resize(im_out, (1920, 1080))               
                cv2.imshow("frame", frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
                
            
        
    # close all resources used      
    cv2.destroyAllWindows()
    try:
        out.release()
    except:
        pass


def draw_track_world(point_array,tf_point_array,background_in,video_in,file_out = None, show = True):    
    """
    combines draw_track and draw_world into a single output video
    """
    # load video file 
    cap = cv2.VideoCapture(video_in)
    assert cap.isOpened(), "Cannot open file \"{}\"".format(video_in)
    
    # load background image 
    world_im = cv2.imread(background_in)
    
    # opens VideoWriter object for saving video file if necessary
    if file_out != None:
        # open video_writer object
        frame_width = int(cap.get(3))+ world_im.shape[1]
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(file_out,cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc('H','2','6','4'), 30, (frame_width,frame_height))
    
    # define random colors for each object
    colormaps = [(random.randrange(0,255),random.randrange(0,255), random.randrange(0,255)) for k in range(0,int(len(point_array[0])/2))]
    
    ret = True
    start = time.time()
    frame_num = 0
    
    # get first frame
    ret, frame = cap.read()
    
    while cap.isOpened():
        
        if ret:
            # get background image
            backg = world_im.copy()
            
            for i in range(0, int(len(point_array[0])/2)):
                # draw points on camera frame
                try:
                    center = (int(point_array[frame_num,i*2]),int(point_array[frame_num,(i*2)+1]))
                    cv2.circle(frame,center, 10, colormaps[i], thickness = -1)
                except:
                    pass # last frame is perhaps not done correctly
                    
                # draw points on world frame
                try:
                    center = (int(tf_point_array[frame_num,i*2]),int(tf_point_array[frame_num,(i*2)+1]))
                    cv2.circle(backg,center, 10, colormaps[i], thickness = -1)
                except:
                    pass # last frame is perhaps not done correctly, may also catch points that fall off image boundary
            
            
            # pad backg image
            bottom_pad = frame_height-backg.shape[0]
            pad = cv2.copyMakeBorder(backg, 0 , bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            # combine two images into a single image
            im_out = np.concatenate((frame,pad),axis = 1)
            
            #summary statistics
            frame_num += 1
            print("FPS of the video is {:5.2f}".format( frame_num / (time.time() - start)))
            
            # get next frame or None
            ret, frame = cap.read()
            
            # save frame to file if necessary
            if file_out != None:
                out.write(im_out)
            
            # output frame
            if show:
                scale = 0.5
                resize = (int(frame_width * scale),int(frame_height*scale))
                im = cv2.resize(im_out, resize)   
                
                cv2.imshow("frame", im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
        
        else:
            break
        
    # close all resources used      
    cap.release()
    cv2.destroyAllWindows()
    try:
        out.release()
    except:
        pass
    
    print("Combination video writing finished.")

def avg_transform_error(orig,trans):
    n_pts = len(orig)
    sum_error = 0
    
    for i in range(0,n_pts):
        x = orig[i,0]
        y = orig[i,0]
        x_true = trans[i,0]
        y_true = trans[i,1]
        error = np.sqrt((x-x_true)**2+(y-y_true)**2)
        sum_error+= error
    return sum_error/n_pts

    
def transform_pt_array(point_array,M):
    """
    Applies 3 x 3  image transformation matrix M to each point stored in the point array
    """
    
    original_shape = point_array.shape
    
    num_points = int(np.size(point_array,0)*np.size(point_array,1)/2)
    # resize array into N x 2 array
    reshaped = point_array.reshape((num_points,2))   
    
    # add third row
    ones = np.ones([num_points,1])
    points3d = np.concatenate((reshaped,ones),1)
    
    # transform points
    tf_points3d = np.transpose(np.matmul(M,np.transpose(points3d)))
    
    # condense to two-dimensional coordinates
    tf_points = np.zeros([num_points,2])
    tf_points[:,0] = tf_points3d[:,0]/tf_points3d[:,2]
    tf_points[:,1] = tf_points3d[:,1]/tf_points3d[:,2]
    
    tf_point_array = tf_points.reshape(original_shape)
    
    return tf_point_array
    
def get_best_transform(x,y):
    """
    given a set of N points in both x and y space, finds the best (lowest avg error)
    transform of 4 points using oppenCV's getPerspectiveTransform
    returns- transformation matrix M
    """
    # test a simple translation
    if False:
        x = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([[1,1],[1,2],[2,1],[2,2]])  
        M_correct = np.array([[1,0,1],[0,1,1],[0,0,1]])
        
    x = np.float32(x)
    y = np.float32(y)
    all_idx = [i for i in range(0,len(x))]
    combos = tuple(combinations(all_idx,4))
    min_err = np.inf
    bestM = 0
    for comb in combos:
         M = cv2.getPerspectiveTransform(x[comb,:],y[comb,:])
         xtf = transform_pt_array(x,M)
         err = avg_transform_error(xtf,y)
         if err < min_err:
             min_err = err
             bestM = M
             bestComb = comb
    return bestM



