from __future__ import division
import torch
import numpy as np
import cv2 

# import detector
from pytorch_yolo_v3.yolo_detector import Darknet_Detector

# import utility functions
from util_detect import detect_video
from util_track import extract_obj_coords
from util_transform import get_best_transform, transform_pt_array, velocities_from_pts, plot_velocities
from util_draw import draw_world, draw_track, draw_track_world


  
if __name__ == "__main__":
    
    
    savenum = 5 # assign unique num to avoid overwriting as necessary
    
    # name in and out files
    video_file = '/home/worklab/Desktop/I24 - test pole visit 5-10-2019/05-10-2019_05-32-15 do not delete/Pelco_Camera_1/capture_008.avi'
    #video_file = '/home/worklab/Desktop/I24 - test pole visit 5-10-2019/axis-ACCC8EB0662C/20190510/08/20190510_084109_D60B_ACCC8EB0662C/20190510_09/20190510_090616_25CE.mkv'
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
    draw_track_world(point_array,tf_points,background_file,detect_file,comb_file,show = True,trail_size = 50)
    
    vel_array = velocities_from_pts(point_array,'im_coord_matching/cam_points2.npy','im_coord_matching/world_feet_points.npy')
    plot_velocities(vel_array,1/30.0)