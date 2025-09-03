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

class MyModel(nn.Sequential):
    def __init__(self, window_size):
        super().__init__()
        self.conv1=nn.Conv2d(3, 64, 3, padding=1)
        self.relu1=nn.ReLU()
        self.maxpooling1=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(64, 128, 3, padding=1)
        self.relu2=nn.ReLU()
        self.maxpooling2=nn.MaxPool2d(2)
        self.conv3=nn.Conv2d(128, 256, 3, padding=1)
        self.relu3=nn.ReLU()
        self.maxpooling3=nn.MaxPool2d(2)
        self.flatten=nn.Flatten()
        self.dense=nn.Linear(256 * (window_size[0]//8) * (window_size[1]//8), 100)
        self.relu4=nn.ReLU()
        self.dense1=nn.Linear(100, 28)
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
        self.paths=[f"{root_images}/{path}" for path in self.paths]

        self._transform=transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        img_path=self.paths[index]
        label=self.points[index]
        if(self._transform):
            image=np.array(PIL.Image.open(img_path))
            old_shape=image.shape
            image=np.array(self._transform(image))
            new_shape=image.shape[1:]
            scale1, scale2=old_shape[0]/new_shape[0], old_shape[1]/new_shape[1]
            label=[(int(x/scale1), int(y/scale2)) for x, y in label]
        else:
            image=np.array(PIL.Image.open(img_path))
        label=np.array(label)
        return image, label
def calculate_mean_std(ds_train):
    mean=0.
    std=0.
    for image, _ in ds_train:
        mean+=image.mean((1, 2))
        std+=image.std((1, 2))
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
    TRANSFORM=T.Compose([
            T.ToTensor(),
            T.Resize(size=WINDOW_SIZE)
        ]   
    )
    ds_train=MyDataset("train", train_img_dir, train_gt, train_fraction=1.0, transform=TRANSFORM)
    MEAN_DATASET, STD_DATASET = calculate_mean_std(ds_train)
    NEW_TRANSFORM=T.Compose([
        T.ToTensor(),
        T.Resize(size=WINDOW_SIZE),
        T.Normalize(MEAN_DATASET, STD_DATASET)
    ]
    )
    ds_train=MyDataset("train", train_img_dir, train_gt,train_fraction=1.0, transform=NEW_TRANSFORM)
    dl_train=data.DataLoader(
        ds_train, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    model=MyModel(WINDOW_SIZE).to(DEVICE)
    loss_fn=torch.nn.MSELoss().to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3)
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
        train_loss=torch.stack(train_loss).mean()
        print(
            f"Epoch {e},",
            f"train_loss: {(train_loss.item()):.8f}",
        )
    return model


def detect(model_filename, test_img_dir):
    WINDOW_SIZE=(100, 100)
    MEAN_DATASET, STD_DATASET = np.array([0.53659433, 0.43041748, 0.37608618], dtype=np.float32), np.array([0.23250134, 0.21281327, 0.20285839], dtype=np.float32)
    TRANSFORM=T.Compose([
        T.ToTensor(),
        T.Resize(size=WINDOW_SIZE),
        T.Normalize(MEAN_DATASET, STD_DATASET)
    ]
    )
    model=MyModel(WINDOW_SIZE)
    model.load_state_dict(torch.load(model_filename, weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    pred={}
    for img_name in sorted(os.listdir(test_img_dir)):
        if not img_name.lower().endswith((".jpg", "jpeg", "png", "bmp")):
            continue
        image_path=os.path.join(test_img_dir, img_name)
        old_shape=np.array(PIL.Image.open(image_path)).shape
        image_tensor=TRANSFORM(PIL.Image.open(image_path)).unsqueeze(0)
        with torch.no_grad():
            pred[img_name] = (model(image_tensor).squeeze()).numpy()
        pred[img_name][::2]*=(old_shape[0]/WINDOW_SIZE[0])
        pred[img_name][1::2]*=(old_shape[1]/WINDOW_SIZE[1])
        pred[img_name]=pred[img_name].astype(np.int64)
    return pred