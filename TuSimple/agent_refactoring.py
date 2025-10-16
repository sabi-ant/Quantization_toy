#########################################################################
##
## train agent that has some utility for training and saving.
##
#########################################################################

import torch.nn as nn
import torch
from util_hourglass import *
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
# from hourglass_network import lane_detection_network
from toy_model.light_toy_model import CustomSeg as lane_detection_network
# from torch.autograd import Function as F
import torch.nn.functional as F
from parameters import Parameters
import math
import util
import hard_sampling
import torch
from torch.ao.quantization import (
  get_default_qconfig_mapping,
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx

############################################################
##
## agent for lane detection
##
############################################################
class Agent(nn.Module):

    #####################################################
    ## Initialize
    #####################################################
    def __init__(self):
        super(Agent, self).__init__()

        self.p = Parameters()

        self.lane_detection_network = lane_detection_network()
        if self.p.qat:
            print("Quantization aware training mode")
            self.lane_detection_network.train()
            qconfig_mapping = get_default_qat_qconfig_mapping("qnnpack")
            self.lane_detection_network = quantize_fx.prepare_qat_fx(self.lane_detection_network, qconfig_mapping, torch.randn(1, 3, 256, 512))


        self.setup_optimizer()

        self.current_epoch = 0

        self.hard_sampling = hard_sampling.hard_sampling()

        print("model parameters: ")
        print(self.count_parameters(self.lane_detection_network))

    def count_parameters(self, model):
	    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def setup_optimizer(self):
        self.lane_detection_optim = torch.optim.Adam(self.lane_detection_network.parameters(),
                                                    lr=self.p.l_rate,
                                                    weight_decay=self.p.weight_decay)

   

    #####################################################
    ## train
    #####################################################
    def train(self, inputs, epoch, gt_point, gt_bin, gt_inst, gt_texture):
        point_loss = self.train_point(inputs, epoch, gt_point, gt_bin, gt_inst, gt_texture)
        return point_loss

    #####################################################
    ## compute loss function and optimize
    #####################################################
    def train_point(self, inputs, epoch, 
                    ground_truth_point, 
                    ground_binary, 
                    ground_truth_instance,
                    gt_texture):
        real_batch_size = inputs.shape[0] #len(target_lanes)

        # update lane_detection_network
        result = self.predict_lanes(inputs)
        lane_detection_loss = 0
        exist_condidence_loss = 0
        nonexist_confidence_loss = 0
        offset_loss = 0
        sisc_loss = 0
        disc_loss = 0
        
        confidance, offset, feature, attribute = result
        # attribute loss
        if 1:
            logits = attribute[:,:,:1].squeeze(-1)  # (batch, 5)
            lane_existance_gt = gt_texture[:,0,:].float()              # (batch, 5)
            lane_existance_loss = F.binary_cross_entropy_with_logits(logits, lane_existance_gt, reduction='mean')
            
            attr_loss = self.focal_loss(attribute[:,:,1:], gt_texture[:,1,:], alpha=0.25, gamma=2.0, reduction='mean', ignore_index=30)
            lane_detection_loss += lane_existance_loss
            lane_detection_loss += attr_loss
        else:
            logits = attribute[:,:,:1].squeeze(-1)  # (batch, 5)
            lane_existance_gt = gt_texture[:,0,:].float()              # (batch, 5)
            lane_existance_loss = F.binary_cross_entropy_with_logits(logits, lane_existance_gt, reduction='mean')
            attr_loss =0

        #compute loss for point prediction

        #exist confidance loss##########################
        #confidance = torch.sigmoid(confidance)
        confidance_gt = ground_truth_point[:, 0, :, :]
        confidance_gt = confidance_gt.view(real_batch_size, 1, self.p.grid_y, self.p.grid_x)
        a = confidance_gt[0][confidance_gt[0]==1] - confidance[0][confidance_gt[0]==1]
        exist_condidence_loss =  exist_condidence_loss +\
            torch.sum( (1-confidance[confidance_gt==1])**2 )/\
            torch.sum(confidance_gt==1)

        #non exist confidance loss##########################
        target = confidance[confidance_gt==0]
        nonexist_confidence_loss =  nonexist_confidence_loss +\
            torch.sum( ( target[target>0.01] )**2 )/\
            (torch.sum(target>0.01)+1)

        #offset loss ##################################
        offset_x_gt = ground_truth_point[:, 1:2, :, :]
        offset_y_gt = ground_truth_point[:, 2:3, :, :]

        predict_x = offset[:, 0:1, :, :]
        predict_y = offset[:, 1:2, :, :]

        offset_loss = offset_loss + \
                    torch.sum( (offset_x_gt[confidance_gt==1] - predict_x[confidance_gt==1])**2 )/\
                    torch.sum(confidance_gt==1) + \
                    torch.sum( (offset_y_gt[confidance_gt==1] - predict_y[confidance_gt==1])**2 )/\
                    torch.sum(confidance_gt==1)

        #compute loss for similarity #################
        feature_map = feature.view(real_batch_size, self.p.feature_size, 1, self.p.grid_y*self.p.grid_x)
        feature_map = feature_map.expand(real_batch_size, self.p.feature_size, self.p.grid_y*self.p.grid_x, self.p.grid_y*self.p.grid_x)#.detach()

        point_feature = feature.view(real_batch_size, self.p.feature_size, self.p.grid_y*self.p.grid_x,1)
        point_feature = point_feature.expand(real_batch_size, self.p.feature_size, self.p.grid_y*self.p.grid_x, self.p.grid_y*self.p.grid_x)#.detach()

        distance_map = (feature_map-point_feature)**2 
        distance_map = torch.sum( distance_map, dim=1 ).view(real_batch_size, 1, self.p.grid_y*self.p.grid_x, self.p.grid_y*self.p.grid_x)

        # same instance
        sisc_loss = sisc_loss+\
            torch.sum(distance_map[ground_truth_instance==1])/\
            torch.sum(ground_truth_instance==1)

        # different instance, same class
        count = (self.p.K1-distance_map[ground_truth_instance==2]) > 0
        count = torch.sum(count).data
        disc_loss = disc_loss + \
            torch.sum((self.p.K1-distance_map[ground_truth_instance==2])[(self.p.K1-distance_map[ground_truth_instance==2]) > 0])/\
            torch.sum(ground_truth_instance==2)

        lane_detection_loss = lane_detection_loss + self.p.constant_exist*exist_condidence_loss
        lane_detection_loss = lane_detection_loss + self.p.constant_nonexist*nonexist_confidence_loss
        lane_detection_loss = lane_detection_loss + self.p.constant_offset*offset_loss
        lane_detection_loss = lane_detection_loss + self.p.constant_alpha*sisc_loss
        lane_detection_loss = lane_detection_loss + self.p.constant_beta*disc_loss + 0.00001*torch.sum(feature**2)
        # lane_detection_loss = lane_detection_loss + self.p.constant_attention*attention_loss

        print("######################################################################")
        print("seg loss")
        print("same instance loss: ", sisc_loss.data)
        print("different instance loss: ", disc_loss.data)

        print("point loss")
        print("exist loss: ", exist_condidence_loss.data)
        print("non-exit loss: ", nonexist_confidence_loss.data)
        print("offset loss: ", offset_loss.data)

        print("--------------------------------------------------------------------")
        print("total loss: ", lane_detection_loss.data)

        self.lane_detection_optim.zero_grad()
        lane_detection_loss.backward()   #divide by batch size
        self.lane_detection_optim.step()

        del confidance, offset, feature
        del ground_truth_point, ground_binary, ground_truth_instance
        del feature_map, point_feature, distance_map
        del exist_condidence_loss, nonexist_confidence_loss, offset_loss, sisc_loss, disc_loss

        # trim = 180 #70+30+70 + 110
        if not self.p.qat:
            trim=50
            if epoch>0 and self.current_epoch != epoch:
                self.current_epoch = epoch
                if 0:
                    if epoch == 30-trim:
                        self.p.l_rate = 0.0005
                        self.setup_optimizer()
                    elif epoch == 60-trim:
                        self.p.l_rate = 0.0002
                        self.setup_optimizer()
                    elif epoch == 90-trim:
                        self.p.l_rate = 0.0001
                        self.setup_optimizer()
                    elif epoch == 100-trim:
                        self.p.l_rate = 0.00005
                        self.setup_optimizer()
                    elif epoch == 110-trim:
                        self.p.l_rate = 0.00002
                        self.setup_optimizer()
                    elif epoch == 160-trim:
                        self.p.l_rate = 0.00001
                        self.setup_optimizer()
                    elif epoch == 190-trim:
                        self.p.l_rate = 0.000005
                        self.setup_optimizer()
                    elif epoch == 220-trim:
                        self.p.l_rate = 0.000001
                        self.setup_optimizer()           
                    elif epoch == 250-trim:
                        self.p.l_rate = 0.0000005
                        self.setup_optimizer()  
                    elif epoch == 280-trim:
                        self.p.l_rate = 0.0000001
                        self.setup_optimizer()  
                    elif epoch == 330-trim:
                        self.p.l_rate = 0.00000001
                        self.setup_optimizer()    
                else:
                    if epoch == 50-trim:
                        self.p.l_rate = 0.000001
                        self.setup_optimizer()
               

        return lane_detection_loss, attr_loss, lane_existance_loss
    
    def focal_loss(self, logits, gt, alpha=0.25, gamma=2.0, reduction='mean', ignore_index=30):
        """
        logits: (batch, 7, 5)
        gt: (batch, 5) long, 값 0~6, 30(ignore)
        """
        # (batch, 7, 5) -> (batch, 5, 7)
        # logits = logits.permute(0, 2, 1)  # (batch, 5, 7)
        gt = gt.long()  # (batch, 5)
        batch, num_pos = gt.shape

        # Flatten for easier indexing
        logits = logits.reshape(-1, logits.size(-1))  # (batch*5, 7)
        gt = gt.reshape(-1)  # (batch*5,)

        # Ignore mask
        valid_mask = (gt != ignore_index)
        logits = logits[valid_mask]
        gt = gt[valid_mask]
        if logits.numel() == 0:
            return logits.sum()*0

        ce_loss = F.cross_entropy(logits, gt, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = alpha * (1 - pt) ** gamma * ce_loss

        if reduction == 'mean':
            return focal.mean()
        elif reduction == 'sum':
            return focal.sum()
        else:
            return focal
    #####################################################
    ## predict lanes
    #####################################################
    def predict_lanes(self, inputs):
        # inputs = torch.from_numpy(inputs).float() 
        # inputs = Variable(inputs).cuda()

        return self.lane_detection_network(inputs)

    #####################################################
    ## predict lanes in test
    #####################################################
    def predict_lanes_test(self, inputs):
        # inputs = torch.from_numpy(inputs).float() 
        # inputs = Variable(inputs).cuda()
        outputs = self.lane_detection_network(inputs)

        return outputs

    #####################################################
    ## Training mode
    #####################################################                                                
    def training_mode(self):
        self.lane_detection_network.train()

    #####################################################
    ## evaluate(test mode)
    #####################################################                                                
    def evaluate_mode(self):
        self.lane_detection_network.eval()

    #####################################################
    ## Setup GPU computation
    #####################################################                                                
    def cuda(self):
        #GPU_NUM = 1
        #device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        #torch.cuda.set_device(device) 
        self.lane_detection_network.cuda()

    #####################################################
    ## Load save file
    #####################################################
    def load_weights(self, epoch, loss):
        self.lane_detection_network.load_state_dict(
            torch.load(self.p.model_path+str(epoch)+'_'+str(loss)+'_'+'lane_detection_network.pkl', map_location='cuda:0'), False
        )

    #####################################################
    ## Save model
    #####################################################
    def save_model(self, epoch, loss):
        torch.save(
            self.lane_detection_network.state_dict(),
            self.p.save_path+str(epoch)+'_'+str(loss)+'_'+'lane_detection_network.pkl'
        )

    def get_data_list(self):
        return self.hard_sampling.get_list()

    def sample_reset(self):
        self.hard_sampling = hard_sampling.hard_sampling()

