'''
a pytorch model to learn motion stereo
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch import Tensor
import cv2
import math
import numpy as np
import time
from numpy.linalg import inv
import torch.onnx 
from easydict import EasyDict as edict

from depth_util import process_camera_parameters, get_pixel_coordinates
from depthNet_model import DepthRefineNet


#Function to Convert to ONNX 
def Convert_ONNX(): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, 6, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "ImageClassifier.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')

if __name__ == "__main__": 

    # Let's build our model 
    #train(5) 
    #print('Finished Training') 

    # Test which classes performed well 
    #testAccuracy() 

    # Let's load the model we just created and test the accuracy per label
    cfg = edict(_run.config) 
    model = DepthRefineNet(idepth_scale=cfg.idepth_scale)
    path = "checkpoint.pt" 
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')), strict=False)
    # model = torch.load(path,map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint['state_dict']) 

    # Test with batch of images 
    #testBatch() 
    # Test how the classes performed 
    #testClassess() 
 
    # Conversion to ONNX 
    Convert_ONNX()
