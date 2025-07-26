#############################################################################################################
##
##  Source code for training. In this source code, there are initialize part, training part, ...
##
#############################################################################################################

import cv2
import torch
# import visdom
#import sys
#sys.path.append('/home/kym/research/autonomous_car_vision/lanedection/code/')
# import agent
import agent_refactoring as agent
import numpy as np
from data_loader import Generator
from parameters import Parameters
import test
import evaluation
import util
import copy
from logger import MLflowLogger
from torch.utils.data import DataLoader
from torch_dataloader import TuSimpleDataset
p = Parameters()

###############################################################
##
## Training
## 
###############################################################
def Training():
    print('Training')

    ####################################################################
    ## Hyper parameter
    ####################################################################
    print('Initializing hyper parameter')

    mlflow_logger = MLflowLogger(run_name="PINet_Training", autostart=True)
    mlflow_logger.start_run()

    #########################################################################
    ## Get dataset
    #########################################################################
    print("Get dataset")
    # train_loader = Generator()
    train_dataset = TuSimpleDataset(mode='train')
    test_dataset = TuSimpleDataset(mode='test')
    train_loader = DataLoader(train_dataset, batch_size=p.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=p.batch_size, shuffle=True, num_workers=4)

    ##############################
    ## Get agent and model
    ##############################
    print('Get agent')
    if p.model_path == "":
        lane_agent = agent.Agent()
    else:
        lane_agent = agent.Agent()

    ##############################
    ## Check GPU
    ##############################
    print('Setup GPU mode')
    if torch.cuda.is_available():
        lane_agent.cuda()
        #torch.backends.cudnn.benchmark=True

    ##############################
    ## Loop for training
    ##############################
    print('Training loop')
    step = 0
    sampling_list = None
    torch.autograd.set_detect_anomaly(True)
    test_data_iterator = iter(test_loader)
    for epoch in range(p.n_epoch):
        lane_agent.training_mode()
        # ground_truth_point, ground_binary, ground_truth_instance
        # for inputs, target_lanes, target_h, test_image, data_list in train_loader.Generate(sampling_list):
        for inputs, gt_point, gt_bin, gt_inst in train_loader:
            print("epoch : " + str(epoch))
            print("step : " + str(step))
            loss_p = lane_agent.train(inputs.float().cuda(), 
                                      epoch, 
                                      gt_point.float().cuda(),
                                      gt_bin.long().cuda(), 
                                      gt_inst.float().cuda())
            torch.cuda.synchronize()
            loss_p = loss_p.cpu().data
            
            if step%200 == 0:
                mlflow_logger.log_metric("loss", loss_p.item(), step=step)
                # lane_agent.save_model(int(step/100), loss_p)
                try:
                    test_image = next(test_data_iterator)
                except StopIteration:
                    test_data_iterator = iter(test_loader) # reset
                    test_image = next(test_data_iterator)
                testing(lane_agent, test_image.float().cuda(), step, loss_p, mlflow_logger)
            step += 1

        lane_agent.sample_reset()

        #evaluation
        if epoch >= 0 and epoch%1 == 0:
            print("evaluation")
            lane_agent.evaluate_mode()
            th_list = [0.8]
            index = [3]

            mlflow_logger.log_model_state_dict(epoch, lane_agent.lane_detection_network, filename=f"model_epoch_{epoch}.pth", artifact_path="models")
        if int(step)>700000:
            break
    mlflow_logger.end_run()

def testing(lane_agent, test_image, step, loss, logger):
    lane_agent.evaluate_mode()

    _, _, ti = test.test(lane_agent, test_image)
    logger.log_image(f"test_image_{step}", ti[0], step=step)
    # cv2.imwrite('test_result/result_'+str(step)+'_'+str(loss)+'.png', ti[0])

    lane_agent.training_mode()

    
if __name__ == '__main__':
    Training()

