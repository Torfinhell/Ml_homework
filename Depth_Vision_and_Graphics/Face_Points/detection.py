import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import PIL.Image
import cv2
from tqdm import tqdm
import math
import torchvision.transforms.v2 as T
from torch.utils import data 
import random
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import typing as tp
from functools import partial

#PARAMETRS
class Config:
    WINDOW_SIZE=(100, 100)
    LAST_LINEAR_SIZE=3800
    BATCH_SIZE=512
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LEFT_RIGHT_PAIRS=[(0,3), (1, 2), (4, 9), (5, 8), (6, 7), (10, 10), (12, 12), (11, 13)]
    MEAN=np.array([129.79718 , 103.865166,  90.321625], dtype=np.float32)
    STD=np.array([65.72256 , 58.388615, 54.779205], dtype=np.float32)
    ROTATE_LIMIT=45
    SCALE_LIMIT=0.05
    SHIFT_LIMIT=0.05
    LEARNING_RATE=8e-3
    ACCUM_STEP=1
    NUM_WORKERS=os.cpu_count()



#-----------------------------------------------------------

#MODEL
class MyModel(nn.Sequential): # TODO Change architechture for somrthing better maybe, maybe read the paper
    def __init__(self, config):
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
        self.dense=nn.Linear( 512* (config.WINDOW_SIZE[0]//16) * (config.WINDOW_SIZE[1]//16), config.LAST_LINEAR_SIZE)
        self.relu5=nn.ReLU()
        self.dropout=nn.Dropout(0.2)
        self.dense1=nn.Linear(config.LAST_LINEAR_SIZE, 28)
    


#-----------------------------------------------------------


#UTILS
def flip_image(image, **kwargs):
    return np.fliplr(image).copy()
def flip_keypoints(points,config, **kwargs):
    points = np.array(points)
    flipped_keys = []

    for p in points:
        x, y = p[0], p[1]
        flipped_x = config.WINDOW_SIZE[1] - x
        flipped_y = y

        rest = p[2:] if len(p) > 2 else []
        flipped_keys.append([flipped_x, flipped_y, *rest])
    for i, j in config.LEFT_RIGHT_PAIRS:
        flipped_keys[i], flipped_keys[j] = flipped_keys[j], flipped_keys[i]

    return np.array(flipped_keys)
def gray2rgb_if_needed(image, **kwargs):
    if image.ndim == 2:  # grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image
def get_info_points(csv_file):
    csv_file=pd.read_csv(csv_file)
    info_points={}
    for _,row in csv_file.iterrows():
        info_points[row.iloc[0]]=np.array(row.iloc[1:])
    return info_points
#-----------------------------------------------------------



#TRANSFORMS
def create_transforms(config,partition:str="train"):
    MEAN_STD=(config.MEAN, config.STD)
    if(partition=="train"):
        flip_keypoints_partial=partial(flip_keypoints, config=config)
        return A.Compose([
             A.Resize(height=config.WINDOW_SIZE[0], width=config.WINDOW_SIZE[1]),
            A.Lambda(image=flip_image, keypoints=flip_keypoints_partial, p=0.5),
            A.ShiftScaleRotate(shift_limit=config.SHIFT_LIMIT, scale_limit=config.SCALE_LIMIT, rotate_limit=config.ROTATE_LIMIT, p=0.5),
            A.Lambda(image=gray2rgb_if_needed),
            A.Normalize(mean=MEAN_STD[0].tolist(), std=MEAN_STD[1].tolist()),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
    return A.Compose([
            A.Resize(height=config.WINDOW_SIZE[0], width=config.WINDOW_SIZE[1]),
            A.Lambda(image=gray2rgb_if_needed),
            A.Normalize(mean=MEAN_STD[0].tolist(), std=MEAN_STD[1].tolist()),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
def reverse_transform(config, ): #for validateMEAN_STD=(config.MEAN, config.STD)
    pass
#-----------------------------------------------------------
#Dataset
from torch.utils import data 
import random
DEFAULT_POINTS=[(0,0) for _ in range(14)]
class MyDataset(data.Dataset):
    
    def __init__(self, 
                 root_images:str, 
                 images_info:dict[str, np.array] | None=None,
                 train_fraction:float=0.8, 
                 split_seed:int=42,
                 transform:tp.Any=None,
                 mode:str="train"):
        super().__init__()
        rng=random.Random(split_seed)
        if(images_info is not None):
            self.paths=sorted(list(images_info.keys()))
            self.shapes=self.get_shapes(root_images)
            self.points=[images_info[path] for path in self.paths]
            self.points=list(map(lambda x: [(x[i], x[i+1]) for i in range(0,len(x), 2)], self.points))
            self.paths, self.shapes, self.points = self.filter(self.paths, self.shapes, self.points)
            combined = list(zip(self.paths, self.shapes, self.points))
            rng.shuffle(combined)
            self.paths, self.shapes, self.points = zip(*combined)
            split_train=int(train_fraction*len(self.paths))
            if(mode=="train"):
                self.paths=self.paths[:split_train]
                self.shapes=self.shapes[:split_train]
                self.points=self.points[:split_train]
            elif(mode=="valid"):
                self.paths=self.paths[split_train:]
                self.shapes=self.shapes[split_train:]
                self.points=self.points[split_train:]
            elif(mode!="all"):
                raise ValueError("Mode is not train or valid or all")
        else:
            self.points=None
            self.paths=sorted([file for file  in os.listdir(root_images) if file.endswith(".jpg")])
            self.shapes=self.get_shapes(root_images)
        self.paths=[f"{root_images}/{file}" for file in self.paths]
        self._transform=transform
    def get_shapes(self, root_images):
        shapes=[]
        for img_path in self.paths:
            shape=np.array(PIL.Image.open(f"{root_images}/{img_path}")).shape
            shapes.append((shape[0], shape[1]))
        return shapes

    def filter(self, paths:tp.List[str], shapes:tp.List[tp.Tuple[int, int]], points:tp.List[tp.List[tp.Tuple[float, float]]]):
        ans = []
        for i, point_group in enumerate(points):
            shape = self.shapes[i]
            is_ok = True
            for x, y in point_group:
                if x > shape[1] or x < 0 or y < 0 or y > shape[0]:
                    is_ok = False
            if is_ok and paths[i].endswith(".jpg"):
                ans.append((paths[i], shapes[i], points[i]))
        return zip(*ans)

    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index:int):
        """
        Takes image (h, w, 3), points array (14, 2)
        Returns tensor image (3,100, 100), and tensor array(14, 2) and label when transform  and file-poinits is specified
        Returns img_path and numpy.image(3, 100, 100) when file_points is not specified 
        """
        img_path=self.paths[index]
        if(self.points is not None):
            label=self.points[index]
            label=np.array(label)
            original_label_size=len(label)
            if(self._transform):
                image=np.array(PIL.Image.open(img_path))
                transformed=None
                image_transformed=image
                label_transformed=label
                while(transformed is None or self.filter([""], [image_transformed.shape], [label_transformed])==((), (), ()) or len(label_transformed)!=original_label_size):
                    transformed=self._transform(image=image, keypoints=label)
                    image_transformed=transformed['image']
                    label_transformed=transformed['keypoints']
                image=image_transformed
                label=label_transformed
            else:
                image=np.array(PIL.Image.open(img_path))
            return image, label
        else:
            if(self._transform):
                image=np.array(PIL.Image.open(img_path))
                transformed=None
                image_transformed=image
                transformed=self._transform(image=image, keypoints=DEFAULT_POINTS)
                image_transformed=transformed['image']
                image=image_transformed
            else:
                image=np.array(PIL.Image.open(img_path))
            return (img_path,image)


#-----------------------------------------------------------





#MAIN_FUNCTIONS
def train_detector(info_points:dict[str, np.array], images_path:str,config=Config(), fast_train:bool=False, save_model_path:str|None=None):
    ''''
    Should read from img_path="./tests/00_test_img_input/train/images"
    points_file="./tests/00_test_img_gt"
    model_path="facepoints_model.pt"
    '''
    if(config.DEVICE==torch.device("cuda:0")):
        torch.cuda.empty_cache()
    if(fast_train):
        num_epochs=1
        config.BATCH_SIZE=8
    else:
        num_epochs=1000
    if(save_model_path is not None):
        os.makedirs(save_model_path, exist_ok=True)
    ds_train=MyDataset(images_path, info_points,mode="train",  transform=create_transforms(config, "train"))
    dl_train=data.DataLoader(
        ds_train, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=config.NUM_WORKERS,
    )
    model=MyModel(config).to(config.DEVICE)
    loss_fn=torch.nn.MSELoss().to(config.DEVICE)#TODO Another Loss
    optimizer=torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    train_losses = []
    best_val_loss=1000
    for e in tqdm(range(num_epochs), total=num_epochs, desc="Training..."):
        model.train()
        train_loss=[]
        optimizer.zero_grad()
        for i, (x_batch, y_batch) in enumerate(dl_train):
            x_batch=x_batch.to(config.DEVICE)
            y_batch=y_batch.to(torch.float32)
            y_batch=y_batch.reshape(y_batch.shape[0], -1).to(config.DEVICE)
            p_batch=model(x_batch)
            loss=loss_fn(p_batch, y_batch)
            loss = loss / config.ACCUM_STEP
            train_loss.append(loss.item())
            loss.backward()
            if((i+1)%config.ACCUM_STEP==0):
                optimizer.step()
                optimizer.zero_grad()
        if (i + 1) % config.ACCUM_STEP == 0:
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        train_loss=sum(train_loss)/len(train_loss)
        train_losses.append(train_loss)
        print(
            f"Epoch {e},",
            f"train_loss: {(train_loss):.8f}",
        )
        # if(e%100==0 or e==num_epochs-1):
        if(e%5==0 and save_model_path is not None):
            model_path=f"{save_model_path}/facepoints_model_check.pt"
            torch.save(model.state_dict(), model_path)
            detected=detect(model_path=model_path, images_path=images_path, config=config)
            ds_valid=MyDataset(images_path, info_points,mode="valid", transform=create_transforms(config, "valid"))
            val_loss_now=evaluate_detect(ds_valid, detected, config)
            if(val_loss_now<best_val_loss):
                model_path=f"{save_model_path}/facepoints_model.pt"
                torch.save(model.state_dict(), model_path)
                best_val_loss=val_loss_now
            print(f"Valid Loss: {val_loss_now}, Best Loss is {best_val_loss}")


def detect(model_path:str, images_path:str, config=Config()):
    model=MyModel(config)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    if(config.DEVICE==torch.device("cuda:0")):
        torch.cuda.empty_cache()
    ds_valid=MyDataset(images_path,images_info=None, mode="all", transform=create_transforms(config, "valid"))
    model.eval()
    ans={}
    for i, (img_path,  image) in enumerate(ds_valid):
        x_batch = image.unsqueeze(0).to(config.DEVICE)
        shape=ds_valid.shapes[i]
        with torch.no_grad():
            p_batch = model(x_batch)
        result_batch=[]
        p_batch=p_batch.detach().cpu().numpy().squeeze()
        for i, (x,y) in enumerate(p_batch.reshape(-1, 2)):
            x*=(shape[1]/config.WINDOW_SIZE[1])
            y*=(shape[0]/config.WINDOW_SIZE[0])
            result_batch.append(x)
            result_batch.append(y)
        ans[os.path.basename(img_path)] =result_batch 
    return ans
    
    


#------------------------------------------------------
#EVALUATION
def evaluate_detect(dataset, detected, config):
    valid_losses=[]
    for ind, (_, label) in enumerate(dataset):
        pred=np.array(detected[os.path.basename(dataset.paths[ind])])
        gt=label.reshape(-1)
        pred_reshaped=[]
        shape=dataset.shapes[ind]
        for i, (x,y) in enumerate(pred.reshape(-1, 2)):
            x/=(shape[1]/config.WINDOW_SIZE[1])
            y/=(shape[0]/config.WINDOW_SIZE[0])
            pred_reshaped.append(x)
            pred_reshaped.append(y)
        loss=np.mean((np.array(pred_reshaped)-np.array(gt))**2)
        valid_losses.append(loss.item())
    return sum(valid_losses)/len(valid_losses)

#------------------------------------------------------------
#MAIN FUNCTION
#TODO Return function to original sizes
if __name__=="__main__":
    #TODO Do changes to MAIN_CONFIG if needed and add config default to functions  and model_path
    image_dir="./tests/00_test_img_input/train/images"#test is the same as train
    file_points="./tests/00_test_img_gt/gt.csv"
    predict_dir="./"
    config=Config()
    model_path="facepoints_model.pt"
    train_detector(get_info_points(file_points), images_path=image_dir, fast_train=False, config=config, save_model_path="models")
    detected=detect(model_path=model_path, images_path=image_dir, config=config)
    ds_valid=MyDataset(image_dir, get_info_points(file_points),mode="valid", transform=create_transforms(config, "valid"))
    evaluate_detect(ds_valid, detected)

