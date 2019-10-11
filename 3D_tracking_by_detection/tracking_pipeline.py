#%% import and definitions
from __future__ import division
import torch
import numpy as np
import cv2 
import _pickle as pickle
import os
import os

os.chdir("..")
# import YOLO detector from pytorch_yolo_v3 submodule
from pytorch_yolo_v3.yolo_detector import Darknet_Detector

# import 3D bbox regressor from torchvision_classifiers submodule
from torchvision_classifiers.trial2_3D_single_class_regression import CNNnet

os.chdir("../KITTI-utils")
# import DepthNet and Track_loader from KITTI_utils module
from util_load import Track_Dataset
from util_tf_net import FcNet

os.chdir("../CV-detection/3D_tracking_by_detection")



def load_detector():
    """
    loads and returns yolo model
    """
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
    
def load_regressor():
    """
    loads and returns model for 3D bounding box regression
    """
    pass

def load_DepthNet():
    """
    loads DepthNet for depth estimation
    """
    pass
    
    
if __name__ == "__main__":    
    #%% load track
    train_im_dir =    "/media/worklab/data_HDD/cv_data/KITTI/Tracking/Tracks/training/image_02"  
    train_lab_dir =   "/media/worklab/data_HDD/cv_data/KITTI/Tracking/Labels/training/label_02"
    train_calib_dir = "/media/worklab/data_HDD/cv_data/KITTI/Tracking/data_tracking_calib(1)/training/calib"
    
    frame_loader = Track_Dataset(train_im_dir,train_lab_dir,train_calib_dir)
    frame_loader.load_track(10)
    
    frame, _ = next(frame_loader)
    
    #%% load yolo model
    detect_net = load_detector()
    
    #%% get all detections first
    all_detections = []
    while frame:
        # get detections
        detections,im_out = detect_net.detect(frame, show = False, verbose = False)
        all_detections.append(detections.cpu().numpy())
        
        # get next frame
        frame,_ = next(frame_loader)
        
    # clean up detections
    detections = remove_duplicates(all_detections)
    detections = condense_detections(detections,style = "SORT_cls")
    
    print("Detection finished.")
    
    #%% release detector model
    del detect_net
    
    #%% track object between frames in 2D context
    objs, point_array = track_SORT(detections,mod_err = 1, meas_err = 10, state_err = 1000, fsld_max = 15)

    
    #%% load regressor model
    regress_net = load_regressor()
    
    #%% ROI align and regress 2D detections into 3D detections
    
    # convert SORT-style into bbox for regressor
    
