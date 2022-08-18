#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from map_boxes import mean_average_precision_for_boxes
import pandas as pd
import random
import argparse
import numpy as np
import cv2
import albumentations as A
from PIL import Image
from PIL import ImageStat
from tqdm import tqdm
import yaml

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='./participants', help='path to championship data')
parser.add_argument('--move_data', type=bool, default=True, help='whether it is necessary to move the data')
parser.add_argument('--augment_data', type=bool, default=False, help='whether it is necessary to augmentation the data')
args = parser.parse_args()

print(f"Количество тренировочных изображений: {len(os.listdir(os.path.join(args.data_dir, 'train/images')))}")
print(f"Количество тестовых изображений: {len(os.listdir(os.path.join(args.data_dir, 'test/images')))}")


x_train_dir = './data_for_yolo/data/images/train'
y_train_dir = './data_for_yolo/data/labels/train'
x_test_dir = './data_for_yolo/data/images/test'
y_test_dir = './data_for_yolo/data/labels/test'
os.makedirs(x_train_dir)
os.mkdir(x_test_dir)
os.makedirs(y_train_dir)
os.mkdir(y_test_dir)

yaml_content = """\
train: ./data/images/train/
val: ./data/images/test/

nc: 5

names: ['car', 'head', 'face', 'human', 'carplate']"""

with open('./data_for_yolo/dataset.yaml', 'w') as f:
    yaml.dump(yaml_content, f)
    

src_x_train_dir = f"{args.data_dir}/train/images"
img_names = os.listdir(src_x_train_dir)

src_y_train_dir = f"{args.data_dir}/train/labels"
label_names  = os.listdir(src_y_train_dir)

dict_ = {'car':0,
         'head': 1, 
         'face': 2,
         'human': 3,
         'carplate':4}


for i, img_name in tqdm(enumerate(img_names), total=len(img_names), desc='Преобразование меток'):
    shutil.copy(os.path.join(src_x_train_dir, img_name), os.path.join(x_train_dir, img_name))
    img_name = img_name[:-4]
    for j, label_name in enumerate(label_names):
        name = '_'.join(label_name.split('_')[:-1])

        if img_name == name:
            with open(f"{y_train_dir}/{'_'.join(label_name.split('_')[:-1])}.txt", 'a') as file:
                with open(f'{src_y_train_dir}/{label_name}', 'r') as txt:
                    s = txt.read()
                    arr = s.replace('0 ', '').replace('\n', ' ')[:-1].split(' ')
                for k in range(len(arr) // 4):
                    file.write(f"{str(dict_[str(label_name.split('_')[-1][:-5])])} {arr[k * 4]} {arr[k * 4 + 1]} {arr[k * 4 + 2]} {arr[k * 4 + 3]}\n")



img_names = [x[:-4] for x in os.listdir(x_train_dir)]
label_names = [x[:-4] for x in os.listdir(y_train_dir)]

val_names = random.sample(img_names, int(len(img_names) * 0.2))

for i, name in tqdm(enumerate(val_names), total=len(val_names), desc='Выделение набора валидации'):
    shutil.move(os.path.join(x_train_dir, name + '.jpg'), os.path.join(x_test_dir, name + '.jpg'))
    shutil.move(os.path.join(y_train_dir, name + '.txt'), os.path.join(y_test_dir, name + '.txt'))
    

# augmentation
category_id_to_name = {0: 'car',
                       1: 'head', 
                       2: 'face',
                       3: 'human',
                       4: 'carplate'}

def brightness(img):
#     im = Image.open(im_file).convert('L')
    im = img.convert('L')
    stat = ImageStat.Stat(im)
    return stat.mean[0]

names = os.listdir(x_train_dir)
if args.augment_data == True:
    for i, name in tqdm(enumerate(names[:]), total=len(names), desc='Аугментация изображений'):
        with open(os.path.join(y_train_dir, name.replace('.jpg', '.txt')), 'r') as txt:
            text = txt.read()

        text = text.split('\n')[:-1]
        text = [s.split(' ') for s in text]

        values = [float(x) if float(x) % 1 != 0 else int(x) for s in text for x in s]        
        values = np.array(values).reshape((len(values) // 5, 5))


        image = cv2.imread(f'{x_train_dir}/{name}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        bboxes = []
        category_ids = []

        PIL_image = Image.fromarray(image.astype('uint8'), 'RGB')
        im_bright = brightness(PIL_image)

        for j, box in enumerate(values):
            category_ids.append(int(box[0]))
            bboxes.append(list(box[1:]))

        num_aug_imgs = 3
        if im_bright < 80:
            transform = A.Compose(
                    [A.RandomBrightnessContrast(brightness_limit=[0.0, 0.4], contrast_limit=[-0.5, 0.5], p=1),
                     A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.1, p=0.7)], 
                    bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
        elif im_bright < 100:
            transform = A.Compose(
                    [A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.4], contrast_limit=[-0.5, 0.5], p=1),
                     A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.1, p=0.7)], 
                    bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
        else:
            transform = A.Compose(
                    [A.RandomBrightnessContrast(brightness_limit=[-0.3, 0.4], contrast_limit=[-0.5, 0.5], p=1),
                     A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.1, p=0.7)], 
                    bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

        if 4 in category_ids:  
            num_aug_imgs = 1

        for k in range(num_aug_imgs):
            random.seed(random.randint(0, 100000))
            transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)

            im = Image.fromarray(transformed['image'])
            im.save(f"{x_train_dir}/{name[:-4]}_{k}.jpg")

            with open(f'{y_train_dir}/{name[:-4]}_{k}.txt', 'w') as txt:
                for cls, (center_x, center_y, w, h) in zip(transformed['category_ids'], transformed['bboxes']):
                    txt.write(f'{cls} {center_x} {center_y} {w} {h}\n')