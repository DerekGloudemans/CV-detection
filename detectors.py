from __future__ import division
import time
import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import cv2 
from pytorch_yolo_v3.util import write_results, load_classes
from pytorch_yolo_v3.darknet import Darknet
from pytorch_yolo_v3.preprocess import prep_image, inp_to_image, letterbox_image



class Darknet_Detector():
    
    def __init__(self, cfgfile, weightfile, class_file ,confidence, nms_threshold,scales, num_classes,resolution):
        self.nms_threshold = nms_threshold
        self.scales = scales
        self.confidence = confidence
        self.num_classes
        self.CUDA = torch.cuda.is_available()
        self. classes = load_classes(class_file)
        
        # initialize darknet
        self.model = Darknet(cfgfile)
        self.model.load_weights(weightfile)
        self.model.net_info["height"] = resolution
        self.inp_dim = int(self.model.net_info["height"])
        
        assert self.inp_dim % 32 == 0 
        assert self.inp_dim > 32
        
        if self.CUDA:
            self.model.cuda()
        
        self.model.eval()
        
    
    # detect objects in one frame
    def detect(self, frame, show = True, verbose = False, write_file = None):
        start = time.time()    
        img, orig_im, dim = prep_image(frame, self.inp_dim)
            
        im_dim = torch.FloatTensor(dim).repeat(1,2)                        
        
        
        if self.CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        
        with torch.no_grad():   
            output = self.model(Variable(img), self.CUDA)
        output = write_results(output, self.confidence, self.num_classes, nms = True, nms_conf = self.nms_theshold)

        
        # rescale bounding bo
        
        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(self.inp_dim/im_dim,1)[0].view(-1,1)
        
        output[:,[1,3]] -= (self.inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (self.inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
        
        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
                
        list(map(lambda x: self.write(x, orig_im), output))
        
        
        cv2.imshow("frame", orig_im)
        print("FPS of the video is {:5.2f}".format( 0.1 / (time.time() - start)))
        
        
    # draws boxes on original image and returns, including label
    def write(self,x, batches, results):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        cls = int(x[-1])
        label = "{0}".format(self.classes[cls])
        #color = random.choice(colors)
        cv2.rectangle(img, c1, c2,'r', 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,'r', -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        return img