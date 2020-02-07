#%% import and definitions
from __future__ import division
import torch
import numpy as np
import cv2 
import _pickle as pickle
import os
import time

from torchvision.ops import roi_align
from torchvision import transforms

os.chdir("..")
# import YOLO detector from pytorch_yolo_v3 submodule
from pytorch_yolo_v3.yolo_detector import Darknet_Detector

from util_detect import remove_duplicates
from util_track import condense_detections, track_SORT, track_naive

os.chdir("torchvision_classifiers")
# import 3D bbox regressor from torchvision_classifiers submodule
from trial2_3D_single_class_regression import CNNnet, plot_batch

os.chdir("../../KITTI-utils")
# import DepthNet and Track_loader from KITTI_utils module
from util_load import Track_Dataset, pil_to_cv, plot_bboxes_3d
from util_tf_net import FcNet
from util_tf_labels import get_image_space_features, im_to_cam_space

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
              'resolution': 512,
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
    
def load_regressor(device):
    """
    loads and returns model for 3D bounding box regression
    """
    
    model = CNNnet()
    model = model.to(device)

    checkpoint_file = "trial2_checkpoint_40.pt"
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_DepthNet(device):
    """
    loads DepthNet for depth estimation
    """
    model = FcNet()
    
    model = model.to(device)
    checkpoint_file = "checkpoints/sigmoid_3d_280.pt"
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
    
    
#%%    
if __name__ == "__main__":    
    #%% load track
    track_num = 10
    train_im_dir =    "/media/worklab/data_HDD/cv_data/KITTI/Tracking/Tracks/training/image_02"  
    train_lab_dir =   "/media/worklab/data_HDD/cv_data/KITTI/Tracking/Labels/training/label_02"
    train_calib_dir = "/media/worklab/data_HDD/cv_data/KITTI/Tracking/data_tracking_calib(1)/training/calib"
    
    frame_loader = Track_Dataset(train_im_dir,train_lab_dir,train_calib_dir)
    frame_loader.load_track(track_num)
    
    frame, _ = next(frame_loader)
    
    #%% load yolo model
    os.chdir("..")
    detect_net = load_detector()
    os.chdir("../CV-detection/3D_tracking_by_detection")

    #%% get all detections first
    start = time.time()
    all_detections = []
    while frame:
        # convert PIL to cv2 image format
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        
        # get detections
        detections,im_out = detect_net.detect(frame, show = False, verbose = False)
        all_detections.append(detections.cpu().numpy())
        
        # get next frame
        frame,_ = next(frame_loader)
        
    # clean up detections
    detections = remove_duplicates(all_detections)
    detections = condense_detections(detections,style = "SORT_cls")
    
    print("Detection finished. ({} sec)".format(time.time() - start))
    
    # release detector model
    del detect_net
    
    #%% track object between frames in 2D context
    start = time.time()
    objs, point_array = track_SORT(detections,mod_err = 10, meas_err = 1, state_err = 1000, fsld_max = 15)
    print("Tracking finished. ({} sec)".format(time.time() - start))
    
    #%% load regressor model
    os.chdir("../torchvision_classifiers")
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    regress_net = load_regressor(device)
    
    frame_loader.load_track(track_num)
    frame, _ = next(frame_loader)
    
    frame_transforms = transforms.Compose([\
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    #%% ROI align and regress 2D detections into 3D detections

    all_3d = []
    start = time.time()
    
    while frame:
        # get bbox coords
        crops = torch.from_numpy(all_detections[frame_loader.cur_frame][:,1:5]).float()
        # add an extra column (first column for roi_align is frame idx)
        crops = torch.cat((torch.zeros(crops.shape[0]).unsqueeze(1),crops),1)
        crops = crops.to(device)
        
        #convert frame to tensor
        frame = frame_transforms(frame).unsqueeze(0).to(device)
        
        # get rois for each object
        obj_crops = roi_align(frame,crops,(224,224))
        
        #regress 3d bbox on each object
        out = regress_net(obj_crops)
        
        # plot first frame
        if frame_loader.cur_frame == 0:
            plot_batch(regress_net,obj_crops)
        
        all_3d.append(out.data.cpu().numpy())
        
        frame, _ = next(frame_loader)
        
    print("3D regression finished. ({} sec)".format(time.time() - start))

    # release regression model
    del regress_net
    
    #%% convert SORT-style into bbox for regressor
    os.chdir("../../KITTI-utils")
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    depth_net = load_DepthNet(device)
    
    P = frame_loader.calib
    im_size = (224,224)
    
    all_labels = []
    for i,frame in enumerate(all_3d):
        label = []
        for j,obj in enumerate(frame):
            # model output depths
            coords = obj.reshape(2,-1)
            
            # add bbox offsets
            coords[0,:] = coords[0,:] + all_detections[i][j,1]
            coords[1,:] = coords[1,:] + all_detections[i][j,2]
            
            X = get_image_space_features(coords,P,im_size)
            X = torch.from_numpy(X).float().to(device)
            pred_depths = (depth_net(X).data.cpu().numpy())[None,:]*100

            # convert into camera space again
            pts_3d = im_to_cam_space(coords,pred_depths,P)
            pts_3d = np.nan_to_num(pts_3d) + 0.0001 # to deal with 0 and nan values
            
            X = np.average(pts_3d[0])
            Y = np.max(pts_3d[1])
            Z = np.average(pts_3d[2])
            
            # find best l,w,h 
            dist = lambda pts,a,b: np.sqrt((pts[0,a]-pts[0,b])**2 + \
                                           (pts[1,a]-pts[1,b])**2 + \
                                           (pts[2,a]-pts[2,b])**2)
            # NOTE - I am not totally sure why the first one is height and not width
            # other than that the points must be in a different order than I suspected
            
            height  = (dist(pts_3d,0,3) + dist(pts_3d,1,2) + \
                      dist(pts_3d,4,7) + dist(pts_3d,5,6)) /4.0
            width  =  (dist(pts_3d,0,1) + dist(pts_3d,3,2) + \
                      dist(pts_3d,4,5) + dist(pts_3d,7,6)) /4.0
            length =  (dist(pts_3d,0,4) + dist(pts_3d,1,5) + \
                      dist(pts_3d,2,6) + dist(pts_3d,3,7)) /4.0
            
            # find best alpha by averaging angles of all 8 relevant line segments
            # defined for line segments backwards to forwards and left to right
            ang = lambda pts,a,b: np.arctan((pts[1,b]-pts[1,a])/(pts[0,b]-pts[0,a]+0.0001))
            angle = (ang(pts_3d,0,1) + ang(pts_3d,3,2) + ang(pts_3d,4,5) + ang(pts_3d,7,6))/4.0 + \
                    ((ang(pts_3d,3,0) + ang(pts_3d,2,1) + ang(pts_3d,7,4) + ang(pts_3d,6,5))/4.0 - np.pi/2)
            alpha = (np.pi - angle)
            if alpha > np.pi/2.0:
                alpha = alpha - np.pi
            # append to new label
            det_dict = {'pos':np.zeros(3),'dim':np.zeros(3)}
            det_dict['pos'][0] = X
            det_dict['pos'][1] = Y
            det_dict['pos'][2] = Z
            det_dict['dim'][0] = height
            det_dict['dim'][1] = width
            det_dict['dim'][2] = length
            det_dict['alpha'] = alpha
            
            # add dummy vals
            det_dict['frame']      = i
            det_dict['id']         = 0
            det_dict['class']      = "Car"
            det_dict['truncation'] = 0
            det_dict['occlusion']  = 0
            det_dict['rot_y'] = 0
            
            label.append(det_dict)
        all_labels.append(label)
    
    
    test = Track_Dataset(train_im_dir,train_lab_dir,train_calib_dir)
    test.load_track(track_num)
    
    
    
    im,_ = next(test)
    frame = 0
    label = all_labels[frame]
    while im:
        
        cv_im = pil_to_cv(im)
        if True:
            cv_im = plot_bboxes_3d(cv_im,label,test.calib)
            #cv_im = plot_bboxes_2d(cv_im,label)
        cv2.imshow("Frame",cv_im)
        key = cv2.waitKey(1) & 0xff
        #time.sleep(1/30.0)
#        if frame > 60:
#            cv2.imwrite("temp{}.png".format(frame),cv_im)
        frame +=1
        
        if key == ord('q'):
            break
        
        # load next frame
        im,label = next(test)
        label = all_labels[frame]

        
    cv2.destroyAllWindows()