from __future__ import division
import time
import torch
import numpy as np
import cv2 
from pytorch_yolo_v3.detector import Darknet_Detector


def detect_video(video_file, detector, verbose = True, show = True, save = False):
    
    # open up a videocapture object
    cap = cv2.VideoCapture(video_file)
    # verify file is opened
    assert cap.isOpened(), "Cannot open file \"{}\"".format(video_file)
    
    # opens VideoWriter object for saving video file if necessary
    if save:
        # open video_writer object
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter("detections_" + video_file,cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc('H','2','6','4'), 30, (frame_width,frame_height))
    
    #main loop   
    start = time.time()
    frames = 0
    ret = True
    all_detections = []
    while cap.isOpened():
        
        if ret:
            # read one frame
            ret, frame = cap.read()
            
            # detect frame
            detections,im_out = detector.detect(frame, show = False, verbose = False)
            if True: # convert to numpy array
                all_detections.append(detections.cpu().numpy())
            #summary statistics
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            
             # save frame to file if necessary
            if save:
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
    matchings = np.zeros(len(first))
    unflat = lambda x: (x%len(second), x //len(second))
    while np.min(dist) < threshold:
        min_f, min_s = unflat(np.argmin(dist))
        matchings[min_f] = min_s
        dist[:,min_f] = np.inf
        dist[:,min_s] = np.inf
        dist[min_f,:] = np.inf
        dist[min_s,:] = np.inf
        
    return matchings

# testing code    
if __name__ == "__main__":
    # loads model unless already loaded
    try:
       net
    except:
        net = Darknet_Detector('pytorch_yolo_v3/cfg/yolov3.cfg','pytorch_yolo_v3/yolov3.weights','pytorch_yolo_v3/data/coco.names',0.5,0.1,80,1024)
        print("Model reloaded.")
    
        # tests that net is working correctly
        test ='pytorch_yolo_v3/imgs/person.jpg'
        out = net.detect(test)
        torch.cuda.empty_cache()    
        
#    video_file = 'capture_005.avi'
#    video_file = '/home/worklab/Desktop/I24 - test pole visit 5-10-2019/05-10-2019_05-32-15 do not delete/Pelco_Camera_1/capture_008.avi'
#    detections = detect_video(video_file,net,show = True, save=False)
#    coords = condense_detections(detections)
        
    
