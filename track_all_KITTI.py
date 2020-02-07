from __future__ import division
import torch
import numpy as np
import cv2 
import _pickle as pickle
import os

# import detector
from pytorch_yolo_v3.yolo_detector import Darknet_Detector

# import utility functions
from util_detect import remove_duplicates, detect_frames
from util_track import track_naive,track_SORT,condense_detections, objs_to_KITTI_text

def load_yolo():
    # loads model unless already loaded
    try:
       net
    except:
        params = {'cfg_file' :'pytorch_yolo_v3/cfg/yolov3.cfg',
                  'wt_file': 'pytorch_yolo_v3/yolov3.weights',
                  'class_file': 'pytorch_yolo_v3/data/coco.names',
                  'pallete_file': 'pytorch_yolo_v3/pallete',
                  'nms_threshold': 0.5,
                  'conf': 0.52,
                  'resolution': 1024,
                  'num_classes': 80}
        
        net = Darknet_Detector(**params)
        print("Model reloaded.")
    
        # tests that net is working correctly
        if False:
            test ='pytorch_yolo_v3/imgs/person.jpg'
            test ='/media/worklab/data_HDD/cv_data/KITTI/Tracking/Tracks/training/image_02/0000/000000.png'
            out = net.detect(test)
            torch.cuda.empty_cache()
    return net


if __name__ == "__main__":
    
    show = True
    
    # name in files
    KITTI_directory = '/media/worklab/data_HDD/cv_data/UA_Detrac/DETRAC-train-data/Insight-MVT_Annotation_Train'
    out_directory = '/media/worklab/data_HDD/temp'

    net = load_yolo()

    dir_list = next(os.walk(KITTI_directory))[1]
    dir_list.sort()
    tracks = [os.path.join(KITTI_directory,item) for item in dir_list]
    
    for i, track_dir in enumerate(tracks):  
        print("On track {}".format(i))
        
        out_path = os.path.join(out_directory,dir_list[i]+ ".txt")
        try:
            f = open(out_path, 'r')
            f.close()
        except:
            # get detections
            detections = detect_frames(track_dir,net, show = True)
            detections = remove_duplicates(detections)
            detections = condense_detections(detections,style = "SORT_cls")
            
            # track objects and draw on video
            objs, point_array = track_SORT(detections,mod_err = 1, meas_err = 10, state_err = 1000, fsld_max = 15)
            
            # pickle objs so text files can be regenerated quickly
            f = open(os.path.join(out_directory,"objs{}".format(i)),'wb')
            pickle.dump(objs,f)
            f.close()
            
            objs_to_KITTI_text(objs,out_path)
        