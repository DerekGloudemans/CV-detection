from __future__ import division
import torch
import numpy as np
import cv2 
from PIL import Image, ImageFile
from torchvision import models, transforms
import torchvision.transforms.functional as TF

# import detector
from pytorch_yolo_v3.yolo_detector import Darknet_Detector
from torchvision_classifiers.split_net_utils import SplitNet, load_model#, monte_carlo_detect

# import utility functions
from util_detect import detect_video, remove_duplicates
from util_track import track_naive,track_SORT,condense_detections,KF_Object, match_hungarian
from util_transform import get_best_transform, transform_pt_array, velocities_from_pts, plot_velocities
from util_draw import draw_world, draw_track, draw_track_world


import time


def monte_carlo_detect(image,model,state_vec,covariance_vec, device, num_samples= 4):
    """
    Generate a random set of windows from original image based on the distributions
    specified by state_vec and covariance_mat for an object
    Then, pass all images in batches through SplitNet model and return box with
    highest confidence in measurement form 
    
    state_vec - 7 or 10 x 1 numpy array, xysr are first 4 values (for one object)
    covariance_mat - 7x7 or 10x10 numpy matrix, where sigma squared xysr are 
        first 4 diagonal values (for one object)
    model - SplitNet object
    image - PIL or similar obj.
    num_samples - specifies the number of windows to generate for the object
    """
    
    # generate window coords
    # minx, miny, maxx, maxy for each window 
    windows = np.zeros([num_samples,4])
    for i in range(num_samples):
        x = np.random.normal(state_vec[0],np.sqrt(covariance_vec[0]))
        y = np.random.normal(state_vec[1],np.sqrt(covariance_vec[1]))
        s = np.random.normal(state_vec[2],np.sqrt(covariance_vec[2]))
        r = np.random.normal(state_vec[3],np.sqrt(covariance_vec[3])/100.0) #right now, r is about 2 orders of magnitude smaller than other state variables
        
        # all windows are square
        if r > 1:
            window_width = s*r
        else:
            window_width = s
        windows[i,0] = x - window_width
        windows[i,1] = y - window_width
        windows[i,2] = x + window_width
        windows[i,3] = y + window_width
        
    # generate windowed images 
    pad_width = 200
    transform = transforms.Compose([\
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ims = []
    for window in windows:
        # scale, normalize, convert to tensor

        # this prevents against cropping off side of image, adjust pad_width if error
        window = np.floor(window) + pad_width
        pad_im = TF.pad(image,pad_width)
        crop_im = TF.crop(pad_im,window[0],window[1],window_width,window_width)
        im = transform(crop_im)
        ims.append(im)
       
    # convert all images to a single tensor     
    ims = torch.stack(ims)
    ims = ims.to(device)
    
    # pass to model
    cls_out, reg_out = model(ims)
    cls_out = cls_out.data.cpu().numpy()
    reg_out = reg_out.data.cpu().numpy()
    torch.cuda.empty_cache()
    
    # find maximum confidence for class 1 in cls
    idx = np.argmax(cls_out[:,1])
    conf = cls_out[idx,1]
    
    # scale back to crop window coords
    box = reg_out[idx] * 224#224*4 - 224*2
    
    # scale to abs img coords - add x min and multiply by factor by which image 
    # was scaled down to fit into 224 window
    box[0] = box[0] * window_width/224 + windows[idx,0] # xmin
    box[1] = box[1] * window_width/224 + windows[idx,1] # ymin
    box[2] = box[2] * window_width/224 + windows[idx,0] # xmax
    box[3] = box[3] * window_width/224 + windows[idx,1] # ymax
    box = box 
    
    im = np.array(image)
    for window in np.round(windows):
        im = cv2.rectangle(im,(1,2),(23,24),(0.8,0.9,0.1),2)    
    im = cv2.rectangle(im,(box[0],box[1]),(box[2],box[3]),(0.8,0.1,0.1),2)
    cv2.imshow("frame",im)
    cv2.waitKey(0)
    
    # convert to xysr
    x = (box[0] + box[2])/2
    y = (box[1] + box[3])/2
    s = box[3]-box[1]
    r = (box[2]-box[0])/s

    return np.array([x,y,s,r,conf])
     
    
    
#----------------------test and implementation code --------------------------#



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
    
    # loads model unless already loaded
    try:
       yolo
       splitnet
    except:
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
        if True:
            test ='pytorch_yolo_v3/imgs/person.jpg'
            out = yolo.detect(test)
            torch.cuda.empty_cache()    
      

        splitnet = SplitNet()
        checkpoint_file = "torchvision_classifiers/checkpoints/7-15-2019-bbox-loss/checkpoint_90.pt"
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
            if frame_num > 5:
                break
            
#             # save frame to file if necessary
#            if save_file != None:
#                out.write(im_out)
#            
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