import cv2
import torch 
import torch.nn as nn
import numpy as np
from ultralytics import YOLO

class LaneSegmentation(nn.Module):
    def __init__(self, checkpoint_path):
        super(LaneSegmentation, self).__init__()
        self.model = YOLO(checkpoint_path)

    def forward(self, x, direction='stright'):
        try:
            output = self.model(x, verbose=False, device=0)
            frame = output[0].orig_img

            cls = output[0].boxes.cls
            img_size = output[0].masks.orig_shape
            mask = torch.zeros((img_size[0], img_size[1], 3)).cuda()
            for i, data in enumerate(output[0].masks.data):
                if cls[i]%2 == 0:
                    mask[:,:,0][data==1] = 100
                else:
                    mask[:,:,1][data==1] = 100
            mask = mask.detach().cpu()
            # cv2.imshow('mask', mask.numpy())
            

            mid_ = int(img_size[1]/2)
            bot_ = img_size[0]
            start_point = (mid_, bot_)

            pos1 = {
                'curve'   : {'left':[], 'right':[],},
                'stright' : {'left':[], 'right':[],}
            }
            pos2 = {'left':[], 'right':[]}
            const = 300
            for idx, xy in enumerate(output[0].masks.xy):
                xy[:,1][xy[:,1]<const]= 0
                xy[:,0][xy[:,1]==0] = 0
                l = len(xy[:,1][xy[:,1]>=const])
                y_pts = xy[:,1][xy[:,1]!=0]
                if len(y_pts) == 0:
                    continue
                point = (xy.sum(0)/l).astype(np.int32)
                y_max = y_pts.max()
                x_max = xy[:,0][xy[:,1]==y_max].mean()
                class_ = cls[idx].item()
                
                x_diff = x_max - mid_
                b_point = (x_max, y_max)
                slope = abs((y_max - point[1])/(x_max - point[0] + 1e-6))
                x_diff2 = b_point[0] - point[0]
                point = np.concatenate([point, [slope]], axis=0) #point, slope, class_
                lane_info = (point, b_point, x_diff2)

                if cls[idx]%2 == 0:
                    if class_ >= 4:
                        pos2['left'].append(np.expand_dims(point, axis=0))
                    else:
                        if x_diff < 0:
                            pos1['curve']['left'].append(lane_info)
                        else:
                            pos1['curve']['right'].append(lane_info)
                else:
                    if class_ >= 4:
                        pos2['left'].append(np.expand_dims(point, axis=0))
                    else:
                        if x_diff < 0:
                            pos1['stright']['left'].append(lane_info)
                        else:
                            pos1['stright']['right'].append(lane_info)
            cnst=0.25
            if direction == 'stright':
                LeftRight = pos1[direction]
                L, R = LeftRight.values()
                if not(len(L)==0 and len(R)==0):

                    for key, vals in LeftRight.items():
                        for val in vals:
                            points, b_point, x_diff2 = val
                            point = points[:2]
                            slope = points[-1]
                            if key=='left':
                                if slope > cnst and x_diff2 < 0:
                                    pos2['left'].append(np.expand_dims(points, axis=0))
                                else:
                                    pos2['right'].append(np.expand_dims(points, axis=0))
                            elif key=='right':
                                if slope > cnst and x_diff2 > 0:
                                    pos2['right'].append(np.expand_dims(points, axis=0))
                                else:
                                    pos2['left'].append(np.expand_dims(points, axis=0))
                else:
                    LeftRight = pos1['curve']
                    for key, vals in LeftRight.items():
                        for val in vals:
                            points, b_point, x_diff2 = val
                            point = points[:2]
                            slope = points[-1]
                            if key=='left':
                                if slope > cnst and x_diff2 < 0:
                                    pos2['left'].append(np.expand_dims(points, axis=0))
                                else:
                                    pos2['right'].append(np.expand_dims(points, axis=0))
                            elif key=='right':
                                if slope > cnst and x_diff2 > 0:
                                    pos2['right'].append(np.expand_dims(points, axis=0))
                                else:
                                    pos2['left'].append(np.expand_dims(points, axis=0))
            else:
                LeftRight = pos1[direction]
                L, R = LeftRight.values()

                if not(len(L)==0 and len(R)==0):

                    for key, vals in LeftRight.items():
                        for val in vals:
                            points, b_point, x_diff2 = val
                            point = points[:2]
                            slope = points[-1]
                            if key=='left':
                                if slope > cnst and x_diff2 < 0:
                                    pos2['left'].append(np.expand_dims(points, axis=0))
                                else:
                                    pos2['right'].append(np.expand_dims(points, axis=0))
                            elif key=='right':
                                if slope > cnst and x_diff2 > 0:
                                    pos2['right'].append(np.expand_dims(points, axis=0))
                                else:
                                    pos2['left'].append(np.expand_dims(points, axis=0))
                else:
                    LeftRight = pos1['stright']
                    for key, vals in LeftRight.items():
                        for val in vals:
                            points, b_point, x_diff2 = val
                            point = points[:2]
                            slope = points[-1]
                            if key=='left':
                                if slope > cnst and x_diff2 < 0:
                                    pos2['left'].append(np.expand_dims(points, axis=0))
                                else:
                                    pos2['right'].append(np.expand_dims(points, axis=0))
                            elif key=='right':
                                if slope > cnst and x_diff2 > 0:
                                    pos2['right'].append(np.expand_dims(points, axis=0))               
                                else:
                                    pos2['left'].append(np.expand_dims(points, axis=0))
            cnst_s1 = 160
            cnst_s2 = 270
            # 빈칸 채우기
            if len(pos2['left'])==0:
                rpoint = np.concatenate(pos2['right'], axis=0)
                rpoint = rpoint[rpoint[:, 1].argmax()][:2]
                diff = rpoint[0] - mid_
                if direction=='stright':
                    if diff < cnst_s1:
                        pos2['left'].append(np.array([[5, rpoint[1]]]))
                    elif diff > cnst_s2:
                        pos2['left'].append(np.array([[mid_, rpoint[1]]]))
                    else:
                        pos2['left'].append(np.array([[img_size[1]-rpoint[0], rpoint[1]+10]]))
                elif direction=='curve':
                    if diff < cnst_s1:
                        pos2['left'].append(np.array([[5, rpoint[1]]]))
                    elif diff > cnst_s2:
                        pos2['left'].append(np.array([[mid_, rpoint[1]]]))
                    else:
                        pos2['left'].append(np.array([[img_size[1]-rpoint[0], rpoint[1]]]))

            elif len(pos2['right'])==0:
                rpoint = np.concatenate(pos2['left'], axis=0)
                rpoint = rpoint[rpoint[:, 1].argmax()][:2]
                diff = mid_ - rpoint[0]
                if direction=='stright':
                    if diff < cnst_s1:
                        pos2['right'].append(np.array([[img_size[1]-5, rpoint[1]]]))
                    elif diff > cnst_s2:
                        pos2['right'].append(np.array([[mid_, rpoint[1]]]))
                    else:
                        pos2['right'].append(np.array([[img_size[1]-rpoint[0], rpoint[1]+10]]))
                elif direction=='curve':

                    if diff < cnst_s1:
                        pos2['right'].append(np.array([[img_size[1]-5, rpoint[1]]]))
                    elif diff > cnst_s2:
                        pos2['right'].append(np.array([[mid_, rpoint[1]]]))
                    else:
                        pos2['right'].append(np.array([[img_size[1]-rpoint[0], rpoint[1]]]))
                
                
            left = np.concatenate(pos2['left'], axis=0)
            right = np.concatenate(pos2['right'], axis=0)
            left = left[left[:, 1].argmax()][:2]
            right = right[right[:, 1].argmax()][:2]
            cv2.circle(frame, (int(left[0]), int(left[1])), 10, (0, 0, 255), -1)
            cv2.circle(frame, (int(right[0]), int(right[1])), 10, (0, 0, 255), -1)
            center = ((left + right)/2).astype(np.int32)
            diff_x = center[0] - mid_

            cv2.arrowedLine(frame, start_point, center,
                                color=(0, 0, 0), 
                                thickness=5, tipLength=0.1)
            cv2.imshow('frame', frame)
            # 방향 return 
            return diff_x
        except:
            pass        