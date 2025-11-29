from torch import nn
import torch
from tqdm.auto import tqdm
from numpy import load, int64
from sklearn.metrics import accuracy_score
from torch import argmax, from_numpy
from copy import deepcopy
from torch.nn.functional import pad
from PIL import Image, ImageDraw
import numpy as np
import os
from collections import defaultdict

# DEVICE=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
DEVICE=torch.device("cpu")

# ============================== 1 Classifier model ============================
def get_cls_model(input_shape=(1, 40, 100)):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channels)
            input shape of image for classification
    :return: nn model for classification
    """
    # your code here \/
    model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),  
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Flatten(),                               
        nn.Linear(32*40*100, 64),                   
        nn.ReLU(),
        nn.Linear(64, 2)                            
    )
    return model


def fit_cls_model(X, y, fast_train=True):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """
    if(fast_train):
        num_epochs=80
    else:
        num_epochs=300
    # your code here \/
    model = get_cls_model()
    X, y=X.to(DEVICE), y.to(DEVICE)
    model.train()
    model=model.to(DEVICE)
    criterion=nn.CrossEntropyLoss().to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(), lr=4e-3, )
    for _ in tqdm(range(num_epochs), total=num_epochs, desc=f"training"):
        optimizer.zero_grad()
        logits=model(X)
        loss=criterion(logits, y.long())
        loss.backward()
        optimizer.step()
    model.eval()
    if(not fast_train):
        torch.save(model.state_dict(), "classifier_model.pt")
    return model.cpu()
    # your code here /\


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    detection_model = deepcopy(cls_model)
    f_idx = next(i for i, l in enumerate(detection_model) if isinstance(l, nn.Flatten))
    detection_model[f_idx] = nn.Identity()  
    linear_idxs = [i for i, l in enumerate(detection_model) if isinstance(l, nn.Linear)]
    l1_idx, l2_idx = linear_idxs
    l1, l2 = detection_model[l1_idx], detection_model[l2_idx]
    channels_input = next(l.out_channels for l in reversed(detection_model[:l1_idx]) if isinstance(l, nn.Conv2d))
    H, W = 40, 100
    conv1 = nn.Conv2d(channels_input, l1.out_features, kernel_size=(H, W))
    conv1.weight.data.copy_(l1.weight.view(l1.out_features, channels_input, H, W))
    conv1.bias.data.copy_(l1.bias.data)
    detection_model[l1_idx] = conv1
    conv2 = nn.Conv2d(l1.out_features, l2.out_features, kernel_size=1)
    conv2.weight.data.copy_(l2.weight.view(l2.out_features, l1.out_features, 1, 1))
    conv2.bias.data.copy_(l2.bias.data)
    detection_model[l2_idx] = conv2
    return detection_model

# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images, visualise_data=False):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    n_rows, n_cols=40, 100
    ans={}
    paths=[path for path in dictionary_of_images]
    images=[dictionary_of_images[path] for  path in paths]
    max_h=max([image.shape[0] for image in images])
    max_w=max([image.shape[1] for image in images])
    tensor_images=torch.stack([pad(torch.from_numpy(image), (0,max_w-image.shape[1], 0, max_h-image.shape[0]), value=0) for image in images])
    pred_images=detection_model(tensor_images[:, None, ...])
    probs = torch.softmax(pred_images, dim=1)
    for i,file in enumerate(paths):
        ans[file]=[]
        if(visualise_data):
            image=Image.fromarray((dictionary_of_images[file]*255).astype(np.uint8)).convert("RGB")
            draw=ImageDraw.Draw(image)
        max_boxes=0
        for row in range(max_h-n_rows+1):
            for col in range(max_w-n_cols+1):
                ans[file].append([row, col , n_rows, n_cols, probs[i, 1, row, col].item()])
                if(visualise_data and ans[file][-1][4]==1.0 and max_boxes<10):
                    top_left = (col, row)
                    bottom_right = (col + n_cols, row + n_rows)
                    draw.rectangle([top_left, bottom_right], outline="red", width=2)
                    max_boxes+=1
        if(visualise_data):
            os.makedirs("visualise", exist_ok=True)
            image.save("visualise/"+file)
    return ans


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    h1, w1, h2, w2=first_bbox[0], first_bbox[1], first_bbox[0]+first_bbox[2], first_bbox[1]+first_bbox[3]
    h1_, w1_, h2_, w2_=second_bbox[0], second_bbox[1], second_bbox[0]+second_bbox[2], second_bbox[1]+second_bbox[3]

    intersect=max(0, min(h2, h2_)-max(h1,h1_))*max(0, min(w2, w2_)-max(w1,w1_))
    union=(h2-h1)*(w2-w1)+(h2_-h1_)*(w2_-w1_)-intersect
    return intersect/union


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes, iou_thr=0.5):
    tps = []
    fps= []
    total_gt = 0

    for filename in pred_bboxes:
        dets = sorted(pred_bboxes[filename], key=lambda x: -x[4])
        gts = [list(gt) for gt in gt_bboxes[filename]]
        total_gt += len(gts)

        for det in dets:
            ious = [calc_iou(det, gt) for gt in gts]
            if not len(ious): 
                fps.append(det)
                continue
            max_idx = ious.index(max(ious))
            if(ious[max_idx]>=iou_thr):
                tps.append(det)
                del gts[max_idx]
            else:
                fps.append(det)
    all=fps+tps
    all.sort(key=lambda x: -x[4])
    tps.sort(key=lambda x: -x[4])
    auc = 0.0
    prev_recall = 0.0
    prev_precision = 1.0
    counter_tp=0
    counter_all=0
    auc=0
    for score in sorted(list(set([score for *_, score in all])),key=lambda x:-x):
        while(counter_tp<len(tps) and tps[counter_tp][4]>=score):
            counter_tp+=1
        while(counter_all<len(all) and all[counter_all][4]>=score):
            counter_all+=1
        recall=counter_tp/total_gt
        precision=counter_tp/counter_all
        assert 0<=recall<=1 and 0<=precision<=1
        auc+=(recall-prev_recall)*(precision+prev_precision)/2
        prev_recall=recall
        prev_precision=precision
    return auc
   
            


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.8):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    ans={}
    for file in detections_dictionary:
        bboxes=sorted(detections_dictionary[file], key=lambda x:-x[4])
        ans[file]=[]
        while(bboxes):
            bbox=bboxes.pop(0)
            for i in range(len(bboxes)-1, -1, -1):
                if(calc_iou(bbox, bboxes[i])>iou_thr):
                    del bboxes[i]
            ans[file].append(bbox)
    return ans
if __name__=="__main__":
    data = load("tests/00_unittest_classifier_input/train_data.npz")
    X = data['X'].reshape(-1, 1, 40, 100)   #pytorch dimensions are (N, C, H, W)
    y = data['y'].astype(int64)

    X, y=from_numpy(X), from_numpy(y)
    cls_model = fit_cls_model(X, y, fast_train=False)
    y_predicted = argmax(cls_model(X), dim = 1)
    print(accuracy_score(y, y_predicted))
