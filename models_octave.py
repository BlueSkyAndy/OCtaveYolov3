from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from PIL import Image

#from utils.parse_config import *
from utils.utils import build_targets
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

from octconv import  *
from utils.fix_utils import *

class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim, exponent):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bce_loss = nn.BCELoss(size_average=True)  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

        self.exponent = exponent

    def forward(self, x, targets=None):
        nA = self.num_anchors
        nB = x.size(0)
        nG = x.size(2)
        stride = self.image_dim / nG

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor


        prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs

        x = prediction[..., 0]  # Center x
        y = prediction[..., 1]  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
        grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
        test_x = True
        #if test_x:
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        #else:
         #   scaled_anchors  = []
         #    for aw,ah in self.anchors:
         #        scaled_anchors.append((aw/float(stride),ah/float(stride)))
         #    scaled_anchors = torch.FloatTensor(scaled_anchors)
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        # pred_boxes[..., 0] = torch.sigmoid(x) + grid_x
        # pred_boxes[..., 1] = torch.sigmoid(y) + grid_y
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        # pred_boxes[..., 2] = torch.exp(w) * anchor_w
        # pred_boxes[..., 3] = torch.exp(h) * anchor_h
        # change e to a small number -> 1.2, for fix pointed need
        pred_boxes[..., 2] = w.data * anchor_w
        pred_boxes[..., 3] = h.data * anchor_h
        # Training
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=pred_conf.cpu().data,
                pred_cls=pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size=nG,
                ignore_thres=self.ignore_thres,
                img_dim=self.image_dim,
                exponent = self.exponent,
            )

            nProposals = int((pred_conf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = float(nCorrect / nProposals) if nProposals > 0 else 0

            # Handle masks
            mask = Variable(mask.type(ByteTensor))
            conf_mask = Variable(conf_mask.type(ByteTensor))

            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = conf_mask - mask

            # Mask outputs to ignore non-existing objects
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(
                pred_conf[conf_mask_true], tconf[conf_mask_true]
            )
            #try:
            #loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss_cls =self.bce_loss(pred_cls[mask], tcls[mask].float())

            #except:
            #    loss_cls = FloatTensor([0])
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )

        else:
            # If not in training phase return predictions
            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes),
                ),
                -1,
            )
            return output


class YoloV3_Branch(nn.Module):
    def __init__(self,input_channel,output_channel,alpha_input=.5,alpha_output=.5,route_layer=None,num_classes=80):
        super(YoloV3_Branch,self).__init__()
        #self.branch_index = branch_index
        self.boxes_attr = num_classes+5

        self.input_channel = input_channel
        self.end_output_channel = self.boxes_attr * 3
        self.stride1_channels = [64,128,256,512,1024]
        self.Last_Conv1=  Last_Conv_BN_ACT(self.stride1_channels[2], self.stride1_channels[3],alpha_input,alpha_output,padding=1)
        self.Last_Conv2 = Last_Conv_BN_ACT(self.stride1_channels[1],self.stride1_channels[2], alpha_input, alpha_output,
                                           padding=1)
        self.module1_up_input =  self.stride1_channels[3]
        self.module1_up_output =  self.stride1_channels[2]
        self.module2_up_input =  self.stride1_channels[2]
        self.module2_up_output =  self.stride1_channels[1]
        self.module1_up = self._make_module_upsample( self.stride1_channels[3], self.stride1_channels[2])#26
        self.module2_up = self._make_module_upsample( self.stride1_channels[2],  self.stride1_channels[1])#52

        self.module0_h = self._make_module_h(input_channel, self.stride1_channels[3])#13
        self.module1_h = self._make_module_h( self.stride1_channels[2]+ self.stride1_channels[3], self.stride1_channels[2])#26
        self.module2_h = self._make_module_h( self.stride1_channels[2] +  self.stride1_channels[1] ,  self.stride1_channels[1])#52

        self.module0_l = self._make_module_l( self.stride1_channels[3], self.end_output_channel)
        self.module1_l = self._make_module_l( self.stride1_channels[2], self.end_output_channel)
        self.module2_l = self._make_module_l( self.stride1_channels[1], self.end_output_channel)


    def _make_module_h(self,input_channel,output_channel):
        layers =[]

        for i in range(0,2) :
            layers.append(nn.Sequential(nn.Conv2d(input_channel,output_channel,1,1,padding=0)))
            layers.append(nn.Sequential(nn.BatchNorm2d(output_channel)))
            layers.append(nn.Sequential(nn.LeakyReLU()))
            layers.append(nn.Sequential(nn.Conv2d(output_channel,input_channel,3,1,padding=1)))
            layers.append(nn.Sequential(nn.BatchNorm2d(input_channel)))
            layers.append(nn.Sequential(nn.LeakyReLU()))

        layers.append(nn.Sequential(nn.Conv2d(input_channel,output_channel,1,1)))
        layers.append(nn.Sequential(nn.BatchNorm2d(output_channel)))
        layers.append(nn.Sequential(nn.LeakyReLU()))

        return nn.Sequential(*layers)

    def _make_module_l(self,input_channel,output_channel):
        layers = []

        layers.append(nn.Sequential(nn.Conv2d(input_channel, input_channel*2, 3, 1, padding=1)))
        layers.append(nn.Sequential(nn.BatchNorm2d(input_channel*2)))
        layers.append(nn.Sequential(nn.LeakyReLU()))
        layers.append(nn.Sequential(nn.Conv2d(input_channel*2, output_channel, 1, 1, padding=0)))

        return nn.Sequential(*layers)

    def _make_module_upsample(self,input_channel,output_channel):

        layers = []
        layers.append(nn.Sequential(nn.Conv2d(input_channel,output_channel,1)))
        layers.append(nn.Sequential(nn.BatchNorm2d(output_channel)))
        layers.append(nn.Sequential(nn.LeakyReLU()))
        layers.append(nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2)))

        return nn.Sequential(*layers)

    def _make_branch_module(self):

        pass


    def forward(self, x,route_1,route_2):

        x = self.module0_h(x)
        out_branch0 = self.module0_l(x)#13

        branch1 =x
        branch1 = self.module1_up(branch1)
        route_1= self.Last_Conv1(route_1) #26
        branch1 = torch.cat((route_1, branch1),1)
        branch1 = self.module1_h(branch1)
        branch2 = branch1

        out_branch1= self.module1_l(branch1)




        branch2 = self.module2_up(branch2)
        route_2 = self.Last_Conv2(route_2)#52
        branch2 = torch.cat((route_2,branch2),1)
        branch2 = self.module2_h(branch2)
        out_branch2 = self.module2_l(branch2)



        return out_branch0,out_branch1,out_branch2





class YoloV3(nn.Module):
    '''
    yolov3 darknet53 octave
    '''
    def __init__(self):
        super(YoloV3,self).__init__()

    def forward(self, x):
        x = []
        return  x

class BottleNeck(nn.Module):
    '''
    residual block
    '''
    def __init__(self,inplane,output=False):

        super(BottleNeck,self).__init__()
        self.outplane = int(inplane/2)
        self.conv1 = Conv_BN_ACT(inplane,self.outplane,1,stride=1,padding=0,activation_layer= nn.LeakyReLU,
                                 alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5)
        self.conv2 = Conv_BN(self.outplane,inplane,3,stride=1,padding=1,
                                 alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5)

        self.relu = nn.LeakyReLU()

        self.output = output

    def forward(self, x):
        identity_h = x[0] if type(x) is tuple else x
        identity_l = x[1] if type(x) is tuple else None

        x_h, x_l = self.conv1(x)
        x_h, x_l = self.conv2((x_h, x_l))

        # if self.downsample is not None:
        #     identity_h, identity_l = self.downsample(x)

        x_h += identity_h
        x_l = x_l + identity_l if identity_l is not None else None

        x_h = self.relu(x_h)
        x_l = self.relu(x_l) if x_l is not None else None
       # print(" out put:{}".format(self.output))
        return x_h, x_l


class DarkNetOctave(nn.Module):
    '''
    input 448*448
    darknet53
    '''
    def __init__(self,layers,num_classes,img_size):
        super(DarkNetOctave , self).__init__()
        self.hyperparams={"learning_rate":0.001}
        self.img_size = img_size
        self.layers = layers
        self.inplane = 32
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu1 = nn.LeakyReLU(0.001)

        self.block_index = 1
        self.stride = 32 # downsample stride
        #self.anchors = [7,12,  13,22,  24,39,  42,59,  51,98,  99,95,  87,162,  161,158,  217,242]
        self.anchors = [7,15,  15,24,  20,39,  33,32,  32,61,  58,56,  52,96,   90,86,    118, 130]
        ###backbone
        self.block1 = self._make_block(64,  layers[0],block_index=1) # 208
        self.block2 = self._make_block(128, layers[1],block_index=2) # 104
        self.block3 = self._make_block(256, layers[2],block_index=3) # 52
        self.block4 = self._make_block(512, layers[3],block_index=4) # 26
        self.block5 = self._make_block(1024,layers[4],block_index=5) # 13
        ###branches
        self.outchannels = [64, 128, 256, 512, 1024]
        self.yolo_branches = YoloV3_Branch(1024,0,num_classes=1)
        ###yolo layers
        self.yolo_layer2 = YOLOLayer([(self.anchors[i],self.anchors[i+1])for i in range(0,6,2)],1,img_size,math.e)#52
        self.yolo_layer1 = YOLOLayer([(self.anchors[i],self.anchors[i+1])for i in range(6,12,2)],1,img_size,math.e)#26
        self.yolo_layer0 = YOLOLayer([(self.anchors[i],self.anchors[i+1])for i in range(12,18,2)],1,img_size,math.e)#13


        self.loss_names = ["all","x", "y", "w", "h", "conf", "cls", "recall", "precision"]
    def forward(self,x,target=None):
        is_training = True if target is not None else None
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.block1(x)
        x = self.block2(x)
        y1 = self.block3(x)  # yolo layer 1 13*13
        y2 = self.block4(y1)  # yolo layer 2 26*26
        y3 = self.block5(y2)  # yolo layer 3 52*52

        branch0,branch1,branch2 = self.yolo_branches(y3,route_1=y2,route_2=y1)# 13,26,52

        self.losses = defaultdict(float)
        loss_for_train = 0.0
        if is_training :
            output0 = self.yolo_layer0(branch0, target)  # 13
            output1 = self.yolo_layer1(branch1, target)  # 26
            output2 = self.yolo_layer2(branch2, target)  # 52


            losses = [output0[i]+output1[i]+output2[i] for i in range(0,len(output2))]

            for name, loss in zip(self.loss_names, losses):
                self.losses[name] += loss

            self.losses["recall"] /= 3
            self.losses["precision"] /= 3

            return  losses[0]
        else:
            output0 = self.yolo_layer0(branch0, target)  # 13
            output1 = self.yolo_layer1(branch1, target)  # 26
            output2 = self.yolo_layer2(branch2, target)  # 52
            return torch.cat((output0,output1,output2),1)

    def _make_block(self,planes,blocks,output=False,block_index = 0):
        '''
        one block represents  the same feature size  module
        :return:
        '''

        layers=[]
        #downsample
        input_channel= self.inplane
        if block_index == 1 :
            downsample= nn.Sequential(First_Conv_BN_ACT(input_channel,planes,3,0.5,0.5,stride=2,padding=1,activation_layer=nn.LeakyReLU))
        else:
            downsample= nn.Sequential(Conv_BN_ACT(input_channel,planes,3,0.5,0.5,stride=2,padding=1,activation_layer=nn.LeakyReLU))

        self.inplane =planes
        layers.append(downsample)
        #block
        for i in range(0,blocks):
            self.block_index+=1

            layers.append(BottleNeck(self.inplane))

        if block_index == 5:
            upsample_size = (int(self.img_size/self.stride),int(self.img_size/self.stride))
            layers.append(nn.Sequential(Last_Conv_BN_ACT(int(self.inplane*0.5),self.inplane,padding=1,upsampled_size=upsample_size)))

        print("block index :%d"%self.block_index)
        return  nn.Sequential(*layers)



















def darnet53_yolo():
    return  DarkNetOctave([1,2,8,8,4],1,416)




def resnet50_yolo():

    pass

###############################################################
import tensorboardX
def test_darknet():
    model = darnet53()

    img = torch.Tensor(1,3,416,416)

    y = model(img)
    with tensorboardX.SummaryWriter("./yolov3_graph/octave") as w:
        w.add_graph(model,img)
    print (y)

if __name__=="__main__":

    test_darknet()
