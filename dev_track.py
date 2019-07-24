from __future__ import division
import torch
import numpy as np
import cv2 
from PIL import Image, ImageFile
from torchvision import models, transforms
import torchvision.transforms.functional as TF

# import detector
from pytorch_yolo_v3.yolo_detector import Darknet_Detector
from torchvision_classifiers.split_net_utils import SplitNet, load_model, plot_batch#, monte_carlo_detect

# import utility functions
from util_detect import detect_video, remove_duplicates
from util_track import track_naive,track_SORT,condense_detections,KF_Object, match_hungarian
from util_transform import get_best_transform, transform_pt_array, velocities_from_pts, plot_velocities
from util_draw import draw_world, draw_track, draw_track_world
import time


if __name__ == "__main__":
    
    # set up a bunch of parameters
    savenum = 3 # assign unique num to avoid overwriting as necessary
    mod_err = 100
    meas_err = 0.01
    state_err = 1 # for KF
    yolo_frequency = 15
    fsld_max = 10
    conf_threshold = 0.7
    
    show = True
    
    #video_file = '/home/worklab/Desktop/I24 - test pole visit 5-10-2019/05-10-2019_05-32-15 do not delete/Pelco_Camera_1/capture_008.avi'
    video_file = '/media/worklab/data_HDD/cv_data/video/traffic_assorted/traffic_0.avi'
    save_file = 'temp_{}.avi'.format(savenum)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache()

    params = {'cfg_file' :'pytorch_yolo_v3/cfg/yolov3.cfg',
              'wt_file': 'pytorch_yolo_v3/yolov3.weights',
              'class_file': 'pytorch_yolo_v3/data/coco.names',
              'pallete_file': 'pytorch_yolo_v3/pallete',
              'nms_threshold': 0.5,
              'conf': 0.52,
              'resolution': 1024,
              'num_classes': 80}
    
    yolo = Darknet_Detector(**params)

    # tests that net is working correctly
    if False:
        test ='pytorch_yolo_v3/imgs/person.jpg'
        out = yolo.detect(test)
        torch.cuda.empty_cache()    
      

    splitnet = SplitNet()
    checkpoint_file = "torchvision_classifiers/splitnet_checkpoint_trained_bboxes.pt"
    checkpoint = torch.load(checkpoint_file)
    splitnet.load_state_dict(checkpoint['model_state_dict'])
    splitnet = splitnet.to(device)
    
    print("Models reloaded.")
    
    
    ############ initialize tracking procedure

    # open up a videocapture object
    cap = cv2.VideoCapture(video_file)
    # verify file is opened
    assert cap.isOpened(), "Cannot open file \"{}\"".format(video_file)
    
    start = time.time()
    frame_num = 0
    active_objs = []
    inactive_objs = []
    
    # get first frame
    ret, frame = cap.read()
    
    # find first frame objects
    detections,im_out = yolo.detect(frame, show = False, verbose = False)
    detections = condense_detections(remove_duplicates(detections.cpu().unsqueeze(0).numpy()),style = "SORT_with_conf")    
    detections = detections[0]
    torch.cuda.empty_cache()
    
    # initialize with all objects found in first frame
    for i,row in enumerate(detections):
        obj = KF_Object(row,0,mod_err,meas_err,state_err)
        obj.all.append(obj.get_xysr_cov()[0])
        active_objs.append(obj)
    
    frame_num += 1
    print("FPS of the video is {:5.2f}".format( 1.0 / (time.time() - start)))
    start = time.time()
    # get second frame
    ret, frame = cap.read()
    
    ################ main loop    
    while cap.isOpened():
        
        if ret:
            
            
            # 1. predict new locations of all objects x_k | x_k-1
            for obj in active_objs:
                obj.predict()
                
            # 2. get new objects using either yolo or splitnet, depending on frame
            
            # object locations according to KF
            locations = np.zeros([len(active_objs),4])
            covs = np.zeros([len(active_objs),4])
            
            for i,obj in enumerate(active_objs):
                locations[i,:],covs[i,:] = obj.get_xysr_cov()
            
            if frame_num % yolo_frequency == 0:
                detections,_ = yolo.detect(frame, show = False, verbose = False)
                detections = condense_detections(remove_duplicates(detections.cpu().unsqueeze(0).numpy()),style = "SORT_with_conf")    
                second = detections[0]
            
                matches = match_hungarian(locations,second)        
                #matches = match_greedy(locations,second)
            else:
                # populate second with detections from splitnet
                second = np.zeros([len(active_objs),5])
                matches = [i for i in range(0,len(locations))]
                pil_im = Image.fromarray(frame)
                for i in range(0,len(active_objs)):
                    
                    second[i,:] = monte_carlo_detect(pil_im,splitnet,locations[i,:],covs[i,:],device,num_samples = 16)
                     # supress match if confidence is too low
                    #if second[i,4] < conf_threshold:
                    #    matches[i] = -1
                second = second[:,:4]
                
               
            # 3. traverse object list
            move_to_inactive = []
            for i in range(0,len(active_objs)):
                obj = active_objs[i]
                
                # update fsld and delete if too high
                if matches[i] == -1:
                    obj.fsld += 1
                    obj.all.append(obj.get_xysr_cov()[0])
                    obj.tags.append(0) # indicates object not detected in this frame
                    if obj.fsld > fsld_max:
                        move_to_inactive.append(i)
                
                else: # object was matched        
                    # update Kalman filter
                    measure_coords = second[matches[i]]
                    obj.update(measure_coords)
                    obj.fsld = 0
                    obj.all.append(obj.get_coords())
                    obj.tags.append(1) # indicates object detected in this frame
    
            # for all unmatched objects, intialize new object
            for j in range(0,len(second)):
                if j not in matches:
                    new_obj = KF_Object(second[j],frame_num,mod_err,meas_err,state_err)
                    new_obj.all.append(new_obj.get_coords())
                    new_obj.tags.append(1) # indicates object detected in this frame
                    active_objs.append(new_obj)
    
            
            # move all necessary objects to inactive list
            move_to_inactive.sort()
            move_to_inactive.reverse()
            for idx in move_to_inactive:
                inactive_objs.append(active_objs[idx])
                del active_objs[idx]
            
            
            
            
            #summary statistics
            frame_num += 1
            print("FPS of the video is {:5.2f}".format( 1.0 / (time.time() - start)))
            start = time.time()
            # get next frame or None
            ret, frame = cap.read()
#            if frame_num > 5:
#                break
            
#             # save frame to file if necessary
#            if save_file != None:
#                out.write(im_out)
            
#            # output frame if necessary
#            if show:
#                im = cv2.resize(im_out, (1920, 1080))               
#                cv2.imshow("frame", im)
#                key = cv2.waitKey(1)
#                if key & 0xFF == ord('q'):
#                    break
#                continue
            
    # close all resources used      
    cap.release()
    cv2.destroyAllWindows()
    try:
        out.release()
    except:
        pass
    torch.cuda.empty_cache()

    # create final data object to be returned
    objs = active_objs + inactive_objs
    
    # create final point array
    points_array = np.zeros([frame_num,len(objs)*2])-1
    for j in range(0,len(objs)):
        obj = objs[j]
        first_frame = int(obj.first_frame)
        for i in range(0,len(obj.all)):
            points_array[i+first_frame,j*2] = obj.all[i][0]
            points_array[i+first_frame,(j*2)+1] = obj.all[i][1]
            #points_array[i+first_frame,(j*4)+2] = obj.all[i][2]
            #points_array[i+first_frame,(j*4)+3] = obj.all[i][3]
            #points_array[i+first_frame,(j*5)+4] = obj.all[i][4]
     
    print("Detection finished")

#draw_track(points_array,video_file, show = True, file_out = "temp1.avi", trail_size = 20)