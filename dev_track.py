from __future__ import division
import torch
import numpy as np
import cv2 
from PIL import Image, ImageFile
from torchvision import models, transforms
import torchvision.transforms.functional as TF

# import detector
from pytorch_yolo_v3.yolo_detector import Darknet_Detector
from torchvision_classifiers.parallel_regression_classification import SplitNet, plot_batch

# import utility functions
from util_detect import detect_video, remove_duplicates
from util_track import track_naive,track_SORT,condense_detections,KF_Object, match_hungarian
from util_transform import get_best_transform, transform_pt_array, velocities_from_pts, plot_velocities
from util_draw import draw_world, draw_track, draw_track_world
import time

def plot_windows(im,windows,show = True):
    """
    plots rectangular windows on a CV2 image.
    im - cv2 image
    windows - num_windows x 4 array where one row is x_min, y_min, x_max, y_max
    """
    
    #im =  cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    for window in windows:
        im = cv2.rectangle(im,(int(window[0]),int(window[1])),(int(window[2]),int(window[3])),(155,155,0),2)
    if show:
        im = cv2.resize(im,(1920,1080))
        cv2.imshow("frame",im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    
    # set up a bunch of parameters
    savenum = 11 # assign unique num to avoid overwriting as necessary
    show = True
    global wer # window expansion ratio
    wer = 5 
    
    # Kalman Filter variables
    mod_err = 1
    meas_err = 3
    state_err = 1
    
    # tracking parameters
    yolo_frequency = 15
    fsld_max = 10
    conf_threshold = 0.5
    window_expand = 0.25
    iou_threshold = 0.3
    proximity_threshold = 0.5
   
    # relevant file paths
    video_file = '/media/worklab/data_HDD/cv_data/video/traffic_assorted/traffic_0.avi'
    intermediate_file = 'pipeline_files/inter_{}.avi'.format(savenum)
    save_file = 'pipeline_files/track_{}.avi'.format(savenum)
    splitnet_checkpoint = "/home/worklab/Documents/Checkpoints/splitnet_centered5_checkpoint_13.pt"
    
    # yolo files and parameters
    params = {'cfg_file' :'pytorch_yolo_v3/cfg/yolov3.cfg',
              'wt_file': 'pytorch_yolo_v3/yolov3.weights',
              'class_file': 'pytorch_yolo_v3/data/coco.names',
              'pallete_file': 'pytorch_yolo_v3/pallete',
              'nms_threshold': 0.5,
              'conf': 0.52,
              'resolution': 1024,
              'num_classes': 80}
    
    # enable CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache()
    
    # load models
    yolo = Darknet_Detector(**params)

    splitnet = SplitNet()
    checkpoint = torch.load(splitnet_checkpoint)
    splitnet.load_state_dict(checkpoint['model_state_dict'])
    splitnet = splitnet.to(device)

    # tests that net is working correctly
    if False:
        test ='pytorch_yolo_v3/imgs/person.jpg'
        out = yolo.detect(test)
        torch.cuda.empty_cache()    
     
    print("Models loaded.")
    
    
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
    
    # opens VideoWriter object for saving video file if necessary
    if intermediate_file != None:
        # open video_writer object
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(intermediate_file,cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc('H','2','6','4'), 30, (frame_width,frame_height))
    
    ################ main loop    
    while cap.isOpened():
        
        if ret:
            
            
            # 1. predict new locations of all objects x_k | x_k-1
            for obj in active_objs:
                obj.predict()
                
            # 2. get new objects using either yolo or splitnet, depending on frame
            # in either case, locations holds states of current frame and second
            # holds states of next frame
            # object locations according to KF
            
            locations = np.zeros([len(active_objs),4])
            covs = np.zeros([len(active_objs),4])
            for i,obj in enumerate(active_objs):
                locations[i,:],covs[i,:] = obj.get_xysr_cov()
            
            # use yolo
            if frame_num % yolo_frequency == 0:
                detections,_ = yolo.detect(frame, show = False, verbose = False)
                detections = condense_detections(remove_duplicates(detections.cpu().unsqueeze(0).numpy()),style = "SORT_with_conf")    
                second = detections[0]
                second = second[:,:-1] # remove confidences
            
                matches = match_hungarian(locations,second,iou_cutoff = iou_threshold)        
                #matches = match_greedy(locations,second)
            
            # use splitnet
            else:
                # populate second with detections from splitnet
                second = np.zeros([len(active_objs),4])
                windows = np.zeros([len(active_objs),4])
                
                matches = [i for i in range(0,len(locations))]
                
                # will hold left_pad, top_pad, and scale factor for each object
                transform_params = np.zeros([len(active_objs),3])
                
                # 2a. for each object, generate window to search in
                for i in range(0, len(locations)):
                    location = locations[i]
                    cov = covs[i]
                    x = location[0]
                    y = location[1]
                    s = location[2] + (cov[0]+cov[1]+cov[2]) * window_expand
                    if s > 2000: # prevents from getting windows that are way too big
                        s = 1000
                    r = location[3]
                    windows[i,0] = int(x - s*r/2)
                    windows[i,2] = int(x + s*r/2)
                    windows[i,1] = int(y - s/2)
                    windows[i,3] = int(y + s/2)
                    
                # 2b. plot windows on the original image to check
                if False:
                    plot_windows(frame.copy(),windows)
                
                #2c. crop these into tensors, scale and normalize, etc.
                pil_im = Image.fromarray(frame)

                # define transforms
                transform = transforms.Compose([\
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
    
                ims = []
                for i,window in enumerate(windows):
                    # crop
                    del_x = window[2]-window[0]
                    del_y = window[3]-window[1]
                    crop_im = TF.crop(pil_im,window[1],window[0],del_y,del_x)
                    
                    # generate pad to make square (alternatively, make a square window)
                    height = crop_im.size[1]
                    width = crop_im.size[0]
                    if height < width:
                        diff = width - height
                        # padding (left, top, right, bottom)\
                        top_padding = int(diff/2+diff%2)
                        transform_params[i,1] = top_padding # save for untransform later
                        padding = (0,top_padding,0,int(diff/2))
                    elif height == width:
                        padding = 0
                    else:
                        diff = height - width
                        left_padding = int(diff/2+diff%2)
                        transform_params[i,0] = left_padding # save for untransform later
                        padding = (left_padding,0,int(diff/2),0)
                    
                    pad_im = TF.pad(crop_im, padding, fill=0, padding_mode='constant')
                                        
                    # scale, normalize and convert to tensor
                    transform_params[i,2] = pad_im.size[0]
                    im = transform(pad_im)
                    ims.append(im)
                   
                # convert all images to a single tensor     
                ims = torch.stack(ims)
                ims = ims.to(device)
                
                # 2d. use the batch_plot to show these tensors, verifying they match up
                if False:
                    plot_batch(splitnet,ims)
                
                # TODO - 2e. pass batch to splitnet
                try:
                    cls_outs, reg_out = splitnet(ims)
                except RuntimeError:
                    print("Divide batch so CUDA does not run out of memory")

                bboxes = reg_out.data.cpu().numpy()
                preds = cls_outs.data.cpu().numpy()
    
                # TODO - 2f. parse back into global image 
                new_windows = np.zeros([len(windows),4])
                for i in range(0,len(bboxes)):
        
                    # transform bbox coords back into im pixel coords
                    bbox = (bboxes[i]* 224*wer - 224*(wer-1)/2).astype(int)
                    
                    # undo scale
                    bbox = (bbox * transform_params[i,2]/224).astype(int)
                    
                    # undo pad and crop
                    bbox[0] = bbox[0] - transform_params[i,0] + windows[i,0]
                    bbox[1] = bbox[1] - transform_params[i,1] + windows[i,1]
                    bbox[2] = bbox[2] - transform_params[i,0] + windows[i,0]
                    bbox[3] = bbox[3] - transform_params[i,1] + windows[i,1]
                    new_windows[i] = bbox
                    
                    # convert back into state for second frame
                    second[i,0] = (bbox[0] + bbox[2])/2 #x center
                    second[i,1] = (bbox[1] + bbox[3])/2 #y center
                    second[i,2] = (bbox[3] - bbox[1]) #s
                    second[i,3] = (bbox[2] - bbox[0])/(second[i,2]+.001) #r
                    

                    
                torch.cuda.empty_cache()
                
                # remove all detections with confidence below 50%
                match_copy = matches.copy()
                keep = []
                for i in range(0,len(second)):
                    if preds[i,1] > conf_threshold:
                        keep.append(i)
                second = second[keep,:]
                # shift matches so there are no matches for removed detections
                removals = 0
                for i in range(0,len(matches)):
                    if i in keep:
                        matches[i] = matches[i] - removals
                    else:
                        matches[i] = -1
                        removals += 1
                
                
                # plot output bboxes on original image to verify correctness
                if True:
                    plot_windows(frame,new_windows,show = not(show))
                
               
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
                    
                    # supress new objects too close to an existing object
                    add = True
                    for obj in active_objs:
                        coords = obj.get_coords()
                        dist = np.sqrt((second[j,0] - coords[0])**2 + (second[j,1] - coords[1])**2)
                        if dist < coords[2]*proximity_threshold:
                            add = False
                            print("Supressed!")
                            break
                        
                    if add:
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
            
            out_im = frame.copy()
            
            #save frame to file if necessary
            if save_file != None:
                out.write(out_im)
            
            # get next frame or None

            ret, frame = cap.read()
            
            # output frame if necessary
            if show:
                cv2.imshow("frame", cv2.resize(out_im, (1920,1080)))
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
     
    print("Detection finished")

draw_track(points_array,intermediate_file, show = False, file_out = save_file, trail_size = 30)