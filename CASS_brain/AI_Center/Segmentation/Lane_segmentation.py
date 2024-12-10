import cv2
import torch 
import torch.nn as nn
import numpy as np
from ultralytics import YOLO


class Centroid():
    def __init__(self):
        self.centroid_x, self.centroid_y = 0, 0

    def get_centroid(self, polygon):
        area = 0
        self.centroid_x = 0
        self.centroid_y = 0
        n = len(polygon)

        for i in range(n):
            j = (i + 1) % n
            factor = polygon[i][0] * polygon[j][1] - polygon[j][0] * polygon[i][1]
            area += factor
            self.centroid_x += (polygon[i][0] + polygon[j][0]) * factor
            self.centroid_y += (polygon[i][1] + polygon[j][1]) * factor
        area /= 2.0
        if area != 0:
            self.centroid_x /= (6 * area)
            self.centroid_y /= (6 * area)


class LaneSegmentation(nn.Module):
    def __init__(self, checkpoint_path):
        super(LaneSegmentation, self).__init__()
        self.model = YOLO(checkpoint_path)
        self.center = Centroid()
        self.N2C = {'0': 'center_road', '1': 'left_road', '2': 'right_road'}

    def forward(self, x, direction='stright'):
        output = self.model(frame, verbose=False, device=0)
        frame = output[0].orig_img

        cls = output[0].boxes.cls
        img_size = output[0].masks.orig_shape

        mid_ = int(img_size[1]/2)
        bot_ = img_size[0]

        mask = torch.zeros((img_size[0], img_size[1], 3)).cuda()
        for i, data in enumerate(output[0].masks.data):
            if 'center_road' in self.N2C[cls[i].item()]:
                mask[:,:,0][data==1] = 100
            elif 'left_road' in self.N2C[cls[i].item()]:
                mask[:,:,1][data==1] = 100
            elif 'right_road' in self.N2C[cls[i].item()]:
                mask[:,:,2][data==1] = 100
                
        mask = mask.detach().cpu()

        roads = {'left':[], 'right':[], 'center':[]}
        for i, xy in enumerate(output[0].masks.xy):
            self.center.get_centroid(xy)
            point = np.array([self.center.centroid_x, center.centroid_y], dtype=np.int32)
            c_name = self.N2C[cls[i].item()]
            if c_name=='center_road':
                roads['center'].append(np.expand_dims(point, axis=0))
            elif c_name=='left_road':
                roads['left'].append(np.expand_dims(point, axis=0))
            elif c_name=='right_road':
                roads['right'].append(np.expand_dims(point, axis=0))
        

        start_point = (mid_, bot_)
        left = roads['left']
        right = roads['right']
        center = roads['center']

        is_left = len(left) != 0
        is_right = len(right) != 0
        is_center = len(center) != 0

        if is_left:
            left = np.concatenate(left, axis=0).mean(0).astype(np.int32)
        if is_right:
            right = np.concatenate(right, axis=0).mean(0).astype(np.int32)
        if is_center:
            center = np.concatenate(center, axis=0)
            center = center[center[:, 0].argsort()]

        if direction == 'left':
            center = center[0]
        elif direction == 'right':
            center = center[-1]
        else:
            if len(center) > 2:
                center = center[1:-1].mean(0).astype(np.int32)
            else:
                center = center[0]        

        cv2.arrowedLine(frame, start_point, center,
                            color=(0, 0, 0), 
                            thickness=5, tipLength=0.1)
        slope = -(start_point[0] - center[0])/(start_point[1] - center[1])
        slope = np.rad2deg(np.arctan(slope)) 
        
        return slope
