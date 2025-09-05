import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PIL.Image
import cv2
from tqdm.auto import tqdm
import math
import torchvision.transforms.v2 as T
from torch.utils import data 
import random
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MyModel(nn.Sequential):
    def __init__(self, window_shape):
        super().__init__()
    
        self.conv1=nn.Conv2d(3, 128, 5, padding=2)
        self.bn1=nn.BatchNorm2d(128)
        self.relu1=nn.ReLU()
        self.maxpooling1=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(128, 256, 3, padding=1)
        self.bn2=nn.BatchNorm2d(256)
        self.relu2=nn.ReLU()
        self.maxpooling2=nn.MaxPool2d(2)
        self.conv3=nn.Conv2d(256, 512, 3, padding=1)
        self.bn3=nn.BatchNorm2d(512)
        self.relu3=nn.ReLU()
        self.maxpooling3=nn.MaxPool2d(2)
        self.conv4=nn.Conv2d(512, 512, 3, padding=1)
        self.bn4=nn.BatchNorm2d(512)
        self.relu4=nn.ReLU()
        self.maxpooling4=nn.MaxPool2d(2)
        self.flatten=nn.Flatten()
        self.dense=nn.Linear( 512* (window_shape[0]//16) * (window_shape[1]//16), 1000)
        self.relu5=nn.ReLU()
        # self.dropout=nn.Dropout(0.1)
        self.dense1=nn.Linear(1000, 28)  

class MyDataset(data.Dataset):
    def __init__(self,
                 mode, 
                 root_images, 
                 images_info,
                 train_fraction=0.8, 
                 split_seed=42,
                 transform=None):
        super().__init__()
        rng=random.Random(split_seed)
        self.paths=list(images_info.keys())
        self.points=[images_info[path] for path in self.paths]
        self.points=list(map(lambda x: [[x[i], x[i+1]] for i in range(0,len(x), 2)], self.points))
        self.paths=[f"{root_images}/{path}" for path in self.paths]
        self.paths, self.points = self.filter(self.paths, self.points)
        combined = list(zip(self.paths, self.points))
        rng.shuffle(combined)
        self.paths, self.points = zip(*combined)
        split_train=int(train_fraction*len(self.paths))
        if(mode=="train"):
            self.paths=self.paths[:split_train]
            self.points=self.points[:split_train]
        elif(mode=="valid"):
            self.paths=self.paths[split_train:]
            self.points=self.points[split_train:]
        else:
            raise ValueError("Mode is not train or valid")

        self._transform=transform
    def filter(self, paths, points):
        ans = []
        for i, point_group in enumerate(points):
            shape = np.array(PIL.Image.open(paths[i])).shape
            is_ok = True
            for x, y in point_group:
                if x > shape[0] or x < 0 or y < 0 or y > shape[1]:
                    is_ok = False
            if is_ok:
                ans.append((paths[i], points[i]))
        return list(zip(*ans))
    def filter_out_of_bound(self, shape, points):
        is_ok = True
        for x, y in points:
            if x > shape[0] or x < 0 or y < 0 or y > shape[1]:
                is_ok = False
        return is_ok

    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        img_path=self.paths[index]
        label=self.points[index]
        original_label_size=len(label)
        if(self._transform):
            image=np.array(PIL.Image.open(img_path))
            transformed=None
            image_transformed=image
            label_transformed=label
            while(transformed is None or not self.filter_out_of_bound(image_transformed.shape,label_transformed) or len(label_transformed)!=original_label_size):
                transformed=self._transform(image=image, keypoints=label)
                image_transformed=transformed['image']
                label_transformed=transformed['keypoints']
            image=image_transformed
            label=label_transformed
        else:
            image=np.array(PIL.Image.open(img_path))
        label=np.array(label)
        return image, label
def calculate_mean_std(ds_train):
    mean=0.
    std=0.
    counter=0
    print(f"Dataset size: {len(ds_train)}")
    for image, _ in ds_train:
        print(f"{counter}/{len(ds_train)}")
        counter+=1
        img = image.float().cpu().numpy()
        img = np.transpose(img, (1,2,0))
        mean += img.mean(axis=(0,1))
        std += img.std(axis=(0,1))
    mean/=len(ds_train)
    std/=len(ds_train)
    return mean, std
def train_detector(train_gt, train_img_dir, fast_train=True):
    if(fast_train):
        NUM_EPOCHS=3
    else:
        NUM_EPOCHS=100
    BATCH_SIZE=32
    WINDOW_SIZE=(100, 100)
    if(torch.cuda.is_available()):
        DEVICE=torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        DEVICE=torch.device("cpu")
    TRANSFORM = A.Compose([
        A.Resize(height=WINDOW_SIZE[0], width=WINDOW_SIZE[1]),
        A.HorizontalFlip(0.5),
        A.ShiftScaleRotate(limit=15, p=0.5),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
    ds_train=MyDataset("train", train_img_dir, train_gt, train_fraction=1.0, transform=TRANSFORM)
    MEAN_DATASET, STD_DATASET = calculate_mean_std(ds_train)
    TRAIN_TRANSFORM= A.Compose([
        A.Resize(height=WINDOW_SIZE[0], width=WINDOW_SIZE[1]),
        A.HorizontalFlip(0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Normalize(mean=MEAN_DATASET.tolist(), std=STD_DATASET.tolist()),
        A.Lambda(image=lambda x, **kwargs: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB) if x.ndim == 2 else x), #for grayscale images convert them to rgb
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
    ds_train=MyDataset("train", train_img_dir, train_gt,train_fraction=1.0, transform=TRAIN_TRANSFORM)
    dl_train=data.DataLoader(
        ds_train, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    model=MyModel(WINDOW_SIZE).to(DEVICE)
    loss_fn=torch.nn.HuberLoss().to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    for e in range(NUM_EPOCHS):
        model.train()
        train_loss=[]
        for x_batch, y_batch in dl_train:
            x_batch=x_batch.to(DEVICE)
            y_batch=y_batch.to(torch.float32)
            y_batch=y_batch.reshape(y_batch.shape[0], -1).to(DEVICE)
            p_batch=model(x_batch)
            loss=loss_fn(p_batch, y_batch)
            train_loss.append(loss.detach())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        train_loss=torch.stack(train_loss).mean()
        print(
            f"Epoch {e},",
            f"train_loss: {(train_loss.item()):.8f}",
        )
    return model


def detect(model_filename, test_img_dir):
    WINDOW_SIZE=(100, 100)
    MEAN_DATASET, STD_DATASET = np.array([136.91046 , 109.890686,  95.62381 ], dtype=np.float32),np.array([60.035656, 55.08362 , 52.576233], dtype=np.float32)
    VALID_TRANSFORM= A.Compose([
        A.Resize(height=WINDOW_SIZE[0], width=WINDOW_SIZE[1]),
        A.HorizontalFlip(0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Normalize(mean=MEAN_DATASET.tolist(), std=STD_DATASET.tolist()),
        A.Lambda(image=lambda x, **kwargs: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB) if x.ndim == 2 else x), #for grayscale images convert them to rgb
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
    model=MyModel(WINDOW_SIZE)
    model.load_state_dict(torch.load(model_filename, weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    pred={}
    for img_name in sorted(os.listdir(test_img_dir)):
        if not img_name.lower().endswith((".jpg", "jpeg", "png", "bmp")):
            continue
        image_path=os.path.join(test_img_dir, img_name)
        old_shape=np.array(PIL.Image.open(image_path)).shape
        image_tensor=VALID_TRANSFORM(PIL.Image.open(image_path)).unsqueeze(0)
        with torch.no_grad():
            pred[img_name] = (model(image_tensor).squeeze()).numpy()
        pred[img_name][::2]*=(old_shape[0]/WINDOW_SIZE[0])
        pred[img_name][1::2]*=(old_shape[1]/WINDOW_SIZE[1])
        pred[img_name]=pred[img_name].astype(np.int64)
    return pred