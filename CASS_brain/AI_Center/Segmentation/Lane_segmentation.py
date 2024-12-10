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

    def forward(self, frame, direction='straight', mode='drive'):
        output = self.model(frame, verbose=False, device=0)
        frame = output[0].orig_img

        cls = output[0].boxes.cls
        img_size = output[0].masks.orig_shape

        mid_ = int(img_size[1]/2)
        bot_ = img_size[0]

        roads = {'center':[], 'side':[]}
        for i, xy in enumerate(output[0].masks.xy):
            self.center.get_centroid(xy)
            point = np.array([self.center.centroid_x, self.center.centroid_y], dtype=np.int32)
            c_name = self.N2C[cls[i].item()]
            if c_name=='center_road':
                roads['center'].append(np.expand_dims(point, axis=0))

            if mode=='left':
                if c_name=='left_road':
                    roads['side'].append(np.expand_dims(point, axis=0))
            elif mode=='right':
                if c_name=='right_road':
                    roads['side'].append(np.expand_dims(point, axis=0))
        

        start_point = (mid_, bot_)

        if mode == 'drive':
            road = roads['center']
            if len(road) == 0:
                road = roads['side']
        else:
            road = roads['side']
            if len(road) == 0:
                road = roads['center']    

        road = np.concatenate(road, axis=0)
        road = road[road[:, 0].argsort()]
        if direction == 'left':
            road = road[0]
        elif direction == 'right':
            road = road[-1]
        else:
            if len(road) > 2:
                road = road[1:-1].mean(0).astype(np.int32)
            else:
                road = road[0]        

        cv2.arrowedLine(frame, start_point, road,
                            color=(0, 0, 0), 
                            thickness=5, tipLength=0.1)

        slope = -(start_point[0] - road[0])/(start_point[1] - road[1])
        slope = np.rad2deg(np.arctan(slope)) 
        return slope
