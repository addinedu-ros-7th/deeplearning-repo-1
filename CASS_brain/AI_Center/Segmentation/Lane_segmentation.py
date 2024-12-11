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

    def forward(self, frame, direction='straight', select_road='center'):
        output = self.model(frame, verbose=False, device=0)
        frame = output[0].orig_img

        cls = output[0].boxes.cls
        img_size = output[0].masks.orig_shape

        mid_ = int(img_size[1]/2)
        bot_ = img_size[0]
        start_point = (mid_, bot_)

        roads = {'center':[], 'side':[]}
        for i, xy in enumerate(output[0].masks.xy):
            self.center.get_centroid(xy)
            point_x = self.center.centroid_x
            point_y = self.center.centroid_y
            slope = -(start_point[0] - point_x)/(start_point[1] - point_y)
            slope = round(np.rad2deg(np.arctan(slope)))
            point = np.array([self.center.centroid_x, self.center.centroid_y, slope], dtype=np.int32)
            c_name = self.N2C[str(int(cls[i].item()))]
            if c_name=='center_road':
                roads['center'].append(np.expand_dims(point, axis=0))

            if select_road=='left':
                if c_name=='left_road':
                    roads['side'].append(np.expand_dims(point, axis=0))
            elif select_road=='right':
                if c_name=='right_road':
                    roads['side'].append(np.expand_dims(point, axis=0))
        
        if select_road == 'center':
            road = roads['center']
            if len(road) == 0:
                road = roads['side']
        else:
            road = roads['side']
            if len(road) == 0:
                road = roads['center']    

        road = np.concatenate(road, axis=0)
        road = road[road[:, -1].argsort()]
        if direction == 'left':
            slope = road[-1]
            road = road[0][:2]
        elif direction == 'right':
            slope = road[-1]
            road = road[-1][:2]
        else:
            if len(road) > 2:
                slope = road[:,-1].mean(0)
                road = road[1:-1][:2].mean(0).astype(np.int32)
            else:
                slope = road[:,-1]
                if slope[0] < -10:
                    road = road[-1][:2] 
                elif slope[-1] > 10:
                    road = road[0][:2]
                else:
                    road = road[0][:2]
            

        cv2.arrowedLine(frame, start_point, road,
                            color=(0, 0, 0), 
                            thickness=5, tipLength=0.1)

        return slope[0]
