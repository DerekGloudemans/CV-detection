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
            all_detections.append(detections)
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

# testing code    
if __name__ == "__main__":
    # loads model unless already loaded
    try:
       net
    except:
        net = Darknet_Detector('pytorch_yolo_v3/cfg/yolov3.cfg','pytorch_yolo_v3/yolov3.weights','pytorch_yolo_v3/data/coco.names',0.5,0.5,80,1024)
        print("Model reloaded.")
    
        # tests that net is working correctly
        test ='pytorch_yolo_v3/imgs/person.jpg'
        out = net.detect(test)
        torch.cuda.empty_cache()    
        
    video_file = 'capture_005.avi'
    detections = detect_video(video_file,net,save=True)