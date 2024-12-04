import os
import json

base_path = 'datasets'
data_name = 'Lane'
image_path = os.path.join(base_path, data_name, 'images')
# json_path = os.path.join(base_path, data_name, 'json')
label_path = os.path.join(base_path, data_name, 'labels')
# json_files = os.listdir(json_path)

# class_dict = {}
# class_list = class_dict.keys()
# i = 0
# k = 0 
# check = False
# for json_file in json_files:
#     with open(os.path.join(json_path, json_file), 'r') as f:
#         json_data = json.load(f)

#     annotations = json_data['annotations']
#     image_name = json_data['image']['file_name']
#     image_size = json_data['image']['image_size']
#     save_path = image_name[:-4] + '.txt'
#     save_path = os.path.join(label_path, save_path)
#     outputs = ''
#     for annotation in annotations:
#         class_type = annotation['class']
#         attributes = annotation['attributes']
#         try:
#             if attributes == []:
#                 check = True
#                 os.remove(os.path.join(image_path, image_name))
#                 os.remove(os.path.join(json_path, json_file))
#                 break
#         except:
#             pass
#         class_name = attributes[0]['value'] + '_' + attributes[1]['value']
        

#         if class_name not in class_list:
#             class_dict[class_name] = str(i)
#             class_list = class_dict.keys()
#             i += 1

#         class_num = class_dict[class_name]
#         data = annotation['data']

#         try:
#             if len(data) <= 2:
#                 check = True
#                 os.remove(os.path.join(image_path, image_name))
#                 os.remove(os.path.join(json_path, json_file))
#                 break
#         except:
#             pass

#         points = ' '
#         for xy in data: 
#             x, y = xy.values()
#             x = x/image_size[1]
#             y = y/image_size[0]
#             points = points + str(x) + ' ' + str(y) + ' '

#         output = class_num + points + '\n'
#         outputs = outputs + output
#     if check:
#         check=False
#         continue

#     with open(save_path, 'a+') as f:
#         f.write(outputs)
    
# for key, value in class_dict.items():
#     print(f'{value} : {key}')

from glob import glob
img_list = glob(os.path.join(image_path, '*.jpg'))
label_list = glob(os.path.join(label_path, '*.txt'))
print(len(img_list), len(label_list))

from sklearn.model_selection import train_test_split
train_img_list, val_img_list = train_test_split(img_list, test_size = 0.3,
random_state = 200)
print(len(train_img_list), len(val_img_list))

with open(os.path.join(base_path, data_name, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_img_list)+'\n')
with open(os.path.join(base_path, data_name, 'val.txt'), 'w') as f:
    f.write('\n'.join(val_img_list)+'\n')