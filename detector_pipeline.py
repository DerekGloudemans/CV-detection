from __future__ import division
import torch
import numpy as np
import cv2 
from pytorch_yolo_v3.yolo_detector import Darknet_Detector
import matplotlib.pyplot as plt
from detector_utils import detect_video, extract_obj_coords, draw_track

  
if __name__ == "__main__":
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
        test ='pytorch_yolo_v3/imgs/person.jpg'
        out = net.detect(test)
        torch.cuda.empty_cache()    
        
    video_file = '/home/worklab/Desktop/I24 - test pole visit 5-10-2019/05-10-2019_05-32-15 do not delete/Pelco_Camera_1/capture_008.avi'
    save_file = 'test_out3.avi'
    final_file = 'test_track3.avi'
    show = True
    
    detections = detect_video(video_file,net,show, save_file=save_file)
    np.save("detections.npy", detections)
    try:
        detections
    except:
        detections = np.load("detections.npy",allow_pickle= True)
    points_array, objs = extract_obj_coords(detections)
    draw_track(points_array,save_file,final_file)
    
    