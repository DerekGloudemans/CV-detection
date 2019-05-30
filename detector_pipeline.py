from __future__ import division
import torch
import numpy as np
import cv2 
from pytorch_yolo_v3.yolo_detector import Darknet_Detector

from detector_utils import detect_video,\
                           extract_obj_coords,\
                           draw_track,\
                           get_best_transform,\
                           transform_pt_array,\
                           draw_world,\
                           draw_track_world

  
if __name__ == "__main__":
    
    
    savenum = 7 # assign unique num to avoid overwriting as necessary
    
    # name in and out files
    video_file = '/home/worklab/Desktop/I24 - test pole visit 5-10-2019/05-10-2019_05-32-15 do not delete/Pelco_Camera_1/capture_008.avi'
    detect_file = 'pipeline_files/detect{}.avi'.format(savenum) 
    track_file = 'pipeline_files/track{}.avi'.format(savenum)
    world_file = 'pipeline_files/world{}.avi'.format(savenum)
    comb_file = 'pipeline_files/comb{}.avi'.format(savenum)
    background_file = 'im_coord_matching/vwd.png'
    show = True
    
    # loads model unless already loaded
    try:
       net
    except:
        params = {'cfg_file' :'pytorch_yolo_v3/cfg/yolov3.cfg',
                  'wt_file': 'pytorch_yolo_v3/yolov3.weights',
                  'class_file': 'pytorch_yolo_v3/data/coco.names',
                  'pallete_file': 'pytorch_yolo_v3/pallete',
                  'nms_threshold': 0.1,
                  'conf': 0.52,
                  'resolution': 1024,
                  'num_classes': 80}
        
        net = Darknet_Detector(**params)
        print("Model reloaded.")
    
        # tests that net is working correctly
        if False:
            test ='pytorch_yolo_v3/imgs/person.jpg'
            out = net.detect(test)
            torch.cuda.empty_cache()    
      
    # get detections
    try:
        detections = np.load("pipeline_files/detections{}.npy".format(savenum),allow_pickle= True)
    except:
        detections = detect_video(video_file,net,show, save_file=detect_file)
        np.save("pipeline_files/detections{}.npy".format(savenum), detections)

    # track objects and draw on video
    point_array, objs = extract_obj_coords(detections)
    #draw_track(point_array,detect_file,track_file,show = True)
    
    # get transform for camera to world space and transform object points
    cam_pts = np.load('im_coord_matching/cam_points2.npy')
    world_pts = np.load('im_coord_matching/world_points2.npy')
    M = get_best_transform(cam_pts,world_pts)
    tf_points = transform_pt_array(point_array,M)
        
    # plot together
    draw_track_world(point_array,tf_points,background_file,detect_file,comb_file,show = True)