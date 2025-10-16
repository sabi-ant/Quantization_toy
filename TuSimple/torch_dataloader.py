#########################################################################
##
##  Data loader source code for TuSimple dataset
##
#########################################################################


import math
import numpy as np
import cv2
import json
import random
from copy import deepcopy
from parameters import Parameters
from torch.utils.data import Dataset
import util


def Translate_Points(point, translation): 
    point = point + translation 
    return point

def Rotate_Points(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

class TuSimpleDataset(Dataset):
    def __init__(self, sampling_list=None, mode='train'):
        self.mode = mode
        self.p = Parameters()
        if self.mode == 'train':
            self.train_data_five = []
            self.train_data_four = []
            self.train_data_three = []
            self.train_data_two = []
            pwd = "."
            with open(f"{pwd}/dataset/five.json") as f:
                self.train_data_five = [json.loads(line) for line in f if line.strip()]
            with open(f"{pwd}/dataset/four.json") as f:
                self.train_data_four = [json.loads(line) for line in f if line.strip()]
            with open(f"{pwd}/dataset/three.json") as f:
                self.train_data_three = [json.loads(line) for line in f if line.strip()]
            with open(f"{pwd}/dataset/two.json") as f:
                self.train_data_two = [json.loads(line) for line in f if line.strip()]
            self.len_image = len(self.train_data_two) + len(self.train_data_three) + len(self.train_data_four) + len(self.train_data_five)
            self.sampling_list = sampling_list
        else:
            self.test_data = []
            with open(self.p.test_root_url+'test_tasks_0627.json') as f:
                self.test_data = [json.loads(line) for line in f if line.strip()]
            self.len_image = len(self.test_data)

    def __len__(self):
        return self.len_image

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.get_data_train()
        else:
            return self.get_data_test()

    def get_data_train(self):
        # 기존 Generator의 Resize_data에서 한 샘플만 추출
        choose = random.random()
        sampling_list = self.sampling_list
        if sampling_list is None or len(sampling_list) < 10:
            if 0.75 <= choose:
                data = random.choice(self.train_data_five)
            elif 0.3 <= choose < 0.75:
                data = random.choice(self.train_data_four)
            elif 0.05 <= choose < 0.3:
                data = random.choice(self.train_data_three)
            else:
                data = random.choice(self.train_data_two)
        else:
            if 0.75 <= choose:
                data = random.choice(self.train_data_five)
            elif 0.35 <= choose < 0.75:
                data = random.choice(self.train_data_four)
            elif 0.2 <= choose < 0.35:
                data = random.choice(self.train_data_three)
            elif 0.15 <= choose < 0.2:
                data = random.choice(self.train_data_two)
            else:
                data = random.choice(sampling_list)

        temp_image = cv2.imread(self.p.train_root_url + data['raw_file'])
        ratio_w = self.p.x_size * 1.0 / temp_image.shape[1]
        ratio_h = self.p.y_size * 1.0 / temp_image.shape[0]
        temp_image = cv2.resize(temp_image, (self.p.x_size, self.p.y_size))
        image = np.rollaxis(temp_image, axis=2, start=0)

        temp_lanes = []
        temp_h = []
        for j in data['lanes']:
            l = np.array(j)
            h = np.array(data['h_samples'])
            l, h = self.make_dense_x(l, h)
            temp_h.append(h * ratio_h)
            temp_lanes.append(l * ratio_w)
        target_lanes = np.array(temp_lanes, dtype=object)
        target_h = np.array(temp_h,dtype=object)
        texture = np.array(data['classes'].split(' '), dtype=int)

        # 데이터 증강 (Flip, Translation, Rotate, Gaussian, Change_intensity, Shadow)
        image, target_lanes, target_h, texture = self.augment(image, target_lanes, target_h, texture)
        ground_truth_point, ground_binary = self.make_ground_truth_point(target_lanes, target_h)
        ground_truth_instance, gt_texture = self.make_ground_truth_instance(target_lanes, target_h, texture)
        return image / 255.0, ground_truth_point, ground_binary, ground_truth_instance, gt_texture
    
    def get_data_test(self):
         #test set image
        test_index = random.randrange(0, self.len_image-1)
        test_image = cv2.imread(self.p.test_root_url+self.test_data[test_index]['raw_file'])
        test_image = cv2.resize(test_image, (self.p.x_size,self.p.y_size))
        return np.rollaxis(test_image/255.0, axis=2, start=0)
    
    def make_dense_x(self, l, h):
        out_x = []
        out_y = []
        p_x = -1
        p_y = -1
        for x, y in zip(l, h):
            if x > 0:
                if p_x < 0:
                    p_x = x
                    p_y = y
                else:
                    out_x.append(x)
                    out_y.append(y)
                    for dense_x in range(min(p_x, x), max(p_x, x), 10):
                        out_x.append(dense_x)
                        if p_x < x:
                            out_y.append(p_y + abs(p_x - dense_x) * abs(p_y - y) / float(abs(p_x - x)))
                        else:
                            out_y.append(p_y + abs(p_x - dense_x) * abs(p_y - y) / float(abs(p_x - x)))
                    p_x = x
                    p_y = y
        return np.array(out_x), np.array(out_y)

    def augment(self, image, target_lanes, target_h, texture):
        # Flip
        if random.random() < self.p.flip_ratio:
            
            image = np.rollaxis(image, axis=2, start=0)
            image = np.rollaxis(image, axis=2, start=0)
            image = cv2.flip(image, 1)
            image = np.rollaxis(image, axis=2, start=0)
            
            for x in target_lanes:
                x[x > 0] = self.p.x_size - x[x > 0]
                x[x < 0] = -2
                x[x >= self.p.x_size] = -2


        # Translation
        if random.random() < self.p.translation_ratio:
            image = np.rollaxis(image, axis=2, start=0)
            image = np.rollaxis(image, axis=2, start=0)
            tx = np.random.randint(-50, 50)
            ty = np.random.randint(-30, 30)
            image = cv2.warpAffine(image, np.float32([[1, 0, tx], [0, 1, ty]]), (self.p.x_size, self.p.y_size))
            image = np.rollaxis(image, axis=2, start=0)
            for x in target_lanes:
                x[x > 0] = x[x > 0] + tx
                x[x < 0] = -2
                x[x >= self.p.x_size] = -2
            for y in target_h:
                y[y > 0] = y[y > 0] + ty

        # Rotate
        if random.random() < self.p.rotate_ratio:
            image = np.rollaxis(image, axis=2, start=0)
            image = np.rollaxis(image, axis=2, start=0)
            angle = np.random.randint(-10, 10)
            M = cv2.getRotationMatrix2D((self.p.x_size // 2, self.p.y_size // 2), angle, 1)
            image = cv2.warpAffine(image, M, (self.p.x_size, self.p.y_size))
            image = np.rollaxis(image, axis=2, start=0)
            for x, y in zip(target_lanes, target_h):
                index_mask = x > 0
                if np.any(index_mask):
                    x[index_mask], y[index_mask] = Rotate_Points(
                        (self.p.x_size // 2, self.p.y_size // 2),
                        (x[index_mask], y[index_mask]),
                        (-angle * 2 * np.pi) / 360
                    )
                    x[x < 0] = -2
                    x[x >= self.p.x_size] = -2
                    x[y < 0] = -2
                    x[y >= self.p.y_size] = -2

        # Gaussian noise
        if random.random() < self.p.noise_ratio:
            img = np.zeros((256, 512, 3), np.uint8)
            m = (0, 0, 0)
            s = (20, 20, 20)
            image = np.rollaxis(image, axis=2, start=0)
            image = np.rollaxis(image, axis=2, start=0)
            cv2.randn(img, m, s)
            image = image + img
            image = np.rollaxis(image, axis=2, start=0)

        # Change intensity
        if random.random() < self.p.intensity_ratio:
            image = np.rollaxis(image, axis=2, start=0)
            image = np.rollaxis(image, axis=2, start=0)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            value = int(random.uniform(-60.0, 60.0))
            if value > 0:
                lim = 255 - value
                v[v > lim] = 255
                v[v <= lim] += value
            else:
                lim = -1 * value
                v[v < lim] = 0
                v[v >= lim] -= lim
            final_hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            image = np.rollaxis(image, axis=2, start=0)

        # Shadow
        if random.random() < self.p.shadow_ratio:
            image = np.rollaxis(image, axis=2, start=0)
            image = np.rollaxis(image, axis=2, start=0)
            top_x, bottom_x = np.random.randint(0, 512, 2)
            coin = 0
            rows, cols, _ = image.shape
            shadow_img = image.copy()
            if coin == 0:
                rand = np.random.randint(2)
                if rand == 0:
                    vertices = np.array([[top_x, 0], [0, 0], [0, rows], [bottom_x, rows]], dtype=np.int32)
                else:
                    vertices = np.array([[top_x, 0], [cols, 0], [cols, rows], [bottom_x, rows]], dtype=np.int32)
                mask = image.copy()
                channel_count = image.shape[2]
                ignore_mask_color = (0,) * channel_count
                cv2.fillPoly(mask, [vertices], ignore_mask_color)
                rand_alpha = np.random.uniform(0.5, 0.75)
                cv2.addWeighted(mask, rand_alpha, image, 1 - rand_alpha, 0., shadow_img)
                shadow_img = np.rollaxis(shadow_img, axis=2, start=0)
                image = shadow_img

        return image, target_lanes, target_h, texture

     #####################################################
    ## Make ground truth for key point estimation
    #####################################################
    def make_ground_truth_point(self, target_lanes, target_h):
        try:
            target_lanes, target_h = util.sort_batch_along_y(target_lanes, target_h)
        except Exception as e:
            print(e)

        ground = np.zeros((3, self.p.grid_y, self.p.grid_x))
        ground_binary = np.zeros((1, self.p.grid_y, self.p.grid_x))

        for lane_index, lane in enumerate(target_lanes):
            for point_index, point in enumerate(lane):
                if point > 0:
                    x_index = int(point/self.p.resize_ratio)
                    y_index = int(target_h[lane_index][point_index]/self.p.resize_ratio)
                    if x_index < 0 or x_index >= self.p.grid_x or y_index < 0 or y_index >= self.p.grid_y:
                        continue
                    try:
                        ground[0][y_index][x_index] = 1.0
                        ground[1][y_index][x_index]= (point*1.0/self.p.resize_ratio) - x_index
                        ground[2][y_index][x_index] = (target_h[lane_index][point_index]*1.0/self.p.resize_ratio) - y_index
                    except Exception as e:
                        print(f"Error at lane_index: {lane_index}, point_index: {point_index}, x_index: {x_index}, y_index: {y_index}")
                        print(e)
                    ground_binary[0][y_index][x_index] = 1

        return ground, ground_binary


    #####################################################
    ## Make ground truth for instance feature
    #####################################################
    def make_ground_truth_instance(self, target_lanes, target_h, texture):

        ground = np.zeros((1, self.p.grid_y*self.p.grid_x, self.p.grid_y*self.p.grid_x))
        temp = np.zeros((1, self.p.grid_y, self.p.grid_x))
        gt_texture = np.zeros((2,5), dtype=int) # row 0: existance, row 1: class
        lane_cluster = 1
        for lane_index, lane in enumerate(target_lanes):
            previous_x_index = 0
            previous_y_index = 0
            for point_index, point in enumerate(lane):
                if point > 0:
                    x_index = int(point/self.p.resize_ratio)
                    y_index = int(target_h[lane_index][point_index]/self.p.resize_ratio)
                    if x_index < 0 or x_index >= self.p.grid_x or y_index < 0 or y_index >= self.p.grid_y:
                        continue
                    temp[0][y_index][x_index] = lane_cluster
                    gt_texture[0][lane_cluster-1] = 1 #existence
                    gt_texture[1][lane_cluster-1] = texture[lane_index]-1 #class
                if previous_x_index != 0 or previous_y_index != 0: #interpolation make more dense data
                    temp_x = previous_x_index
                    temp_y = previous_y_index
                    while False:
                        delta_x = 0
                        delta_y = 0
                        temp[0][temp_y][temp_x] = lane_cluster
                        if temp_x < x_index:
                            temp[0][temp_y][temp_x+1] = lane_cluster
                            delta_x = 1
                        elif temp_x > x_index:
                            temp[0][temp_y][temp_x-1] = lane_cluster
                            delta_x = -1
                        if temp_y < y_index:
                            temp[0][temp_y+1][temp_x] = lane_cluster
                            delta_y = 1
                        elif temp_y > y_index:
                            temp[0][temp_y-1][temp_x] = lane_cluster
                            delta_y = -1
                        temp_x += delta_x
                        temp_y += delta_y
                        if temp_x == x_index and temp_y == y_index:
                            break
                if point > 0:
                    previous_x_index = x_index
                    previous_y_index = y_index

            
            lane_cluster += 1

        for i in range(self.p.grid_y*self.p.grid_x): #make gt
            temp = temp[temp>-1]
            gt_one = deepcopy(temp)
            if temp[i]>0:
                gt_one[temp==temp[i]] = 1   #same instance
                if temp[i] == 0:
                    gt_one[temp!=temp[i]] = 3 #different instance, different class
                else:
                    gt_one[temp!=temp[i]] = 2 #different instance, same class
                    gt_one[temp==0] = 3 #different instance, different class
                ground[0][i] += gt_one
        gt_texture[1][lane_cluster-1:] = 30
        return ground, gt_texture

# 사용 예시:
# from torch.utils.data import DataLoader
# train_dataset = TuSimpleDataset()
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
# for images, lanes, hs, data in train_loader:
#     # 학습 코드 작성
