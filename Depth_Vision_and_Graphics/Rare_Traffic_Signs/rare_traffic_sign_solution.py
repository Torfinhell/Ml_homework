# -*- coding: utf-8 -*-
import csv
import json
import os
import pickle
import random
import shutil
import typing
from concurrent.futures import ProcessPoolExecutor
from torch import nn
import albumentations as A
import lightning as L
import numpy as np
import scipy
import skimage
import skimage.filters
import skimage.io
import skimage.transform
import torch
import torchvision
import tqdm
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
import pandas as pd
from pathlib import Path
# !Этих импортов достаточно для решения данного задания


CLASSES_CNT = 205
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LEARNING_RATE=4e-3
BATCH_SIZE=8
NUM_WORKERS=8
LABEL_SMOOTHING=0.1
WIDTH=30
HEIGHT=30
GAMMA=0.8
MAX_EPOCHS=30


class DatasetRTSD(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения датасета.

    :param root_folders: список путей до папок с данными
    :param path_to_classes_json: путь до classes.json
    """

    def __init__(
        self,
        root_folders: typing.List[str],
        path_to_classes_json: str,
    ) -> None:
        super().__init__()
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)
        ### YOUR CODE HERE - список пар (путь до картинки, индекс класса)
        dirs_classes=[(os.path.join(root_path, dir_class), self.class_to_idx[dir_class]) for root_path in root_folders for dir_class in os.listdir(root_path)]
        self.samples = [(os.path.join(path_dir, file), class_ind) for path_dir, class_ind in dirs_classes for file in os.listdir(path_dir)]
        ### YOUR CODE HERE - cловарь из списков картинок для каждого класса, classes_to_samples[индекс класса] = [список чисел-позиций картинок в self.samples]
        self.classes_to_samples =  {i:[] for i in range(len(self.classes))}
        for i, (_, ind) in enumerate(self.samples):
            self.classes_to_samples[ind].append(i)
        # mean, std=self.calculate_mean_std() #calculate mean std for the dataset
        mean, std=np.array([106.13662484,  88.12364826,  84.8854233]), np.array([35.645314, 35.73020842, 38.32509976])
        ### YOUR CODE HERE - аугментации + нормализация + ToTensorV2
        self.transform = A.Compose([
            A.Resize(width=WIDTH, height=HEIGHT),
            A.Normalize(mean=mean, std=std),
            A.ToTensorV2()
        ])
    def calculate_mean_std(self):
        std=[]
        mean=[]
        for image, _, _ in self:
            mean.append(image.mean(axis=(0, 1)))
            std.append(image.std(axis=(0, 1)))
        return np.array(mean).mean(axis=0), np.array(std).mean(axis=0)
    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int]:
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        img_path, class_ind=self.samples[index]
        image=np.array(Image.open(img_path).convert("RGB"))
        if getattr(self, "transform", None) is not None:
            image=self.transform(image=image)["image"]
        return image, img_path, class_ind

    @staticmethod
    def get_classes(
        path_to_classes_json,
    ) -> typing.Tuple[typing.List[str], typing.Mapping[str, int]]:
        """
        Считывает из classes.json информацию о классах.

        :param path_to_classes_json: путь до classes.json
        """
        with open(path_to_classes_json, "r") as file:
            classes_json=json.load(file)
        class_to_idx={class_:classes_json[class_]["id"] for class_ in classes_json}
        ### YOUR CODE HERE - словарь, class_to_idx['название класса'] = индекс
        ### YOUR CODE HERE - массив, classes[индекс] = 'название класса'
        classes = [class_ for class_, _ in sorted(class_to_idx.items(),key=lambda x:x[1])]
        return classes, class_to_idx

    def __len__(self) -> int:
        """
        Возвращает размер датасета (количество сэмплов).
        """
        return len(self.samples)


class TestData(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения тестового датасета.

    :param root: путь до папки с картинками знаков
    :param path_to_classes_json: путь до classes.json
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """

    def __init__(
        self,
        root: str,
        path_to_classes_json: str,
        annotations_file: str = None,
    ) -> None:
        super().__init__()
        self.root = root
        ### YOUR CODE HERE - список путей до картинок
        self.samples = os.listdir(self.root)
        ### YOUR CODE HERE - преобразования: ресайз + нормализация + ToTensorV2
        with open(path_to_classes_json, "r") as file:
            classes_json=json.load(file)
        self.class_to_idx={class_:classes_json[class_]["id"] for class_ in classes_json}
        self.rare_gt={classes_json[class_]["id"] for class_ in classes_json if classes_json[class_]["type"]=="rare"}
        self.freq_gt={classes_json[class_]["id"] for class_ in classes_json if classes_json[class_]["type"]=="freq"}
        self.classes = [class_ for class_, _ in sorted(self.class_to_idx.items(),key=lambda x:x[1])]
        mean, std=np.array([106.13662484,  88.12364826,  84.8854233]), np.array([35.645314, 35.73020842, 38.32509976])
        self.transform = A.Compose([
            A.Resize(width=WIDTH, height=HEIGHT),
            A.Normalize(mean=mean, std=std),
            A.ToTensorV2()
        ])
        self.targets = None
        if annotations_file is not None:
            ### YOUR CODE HERE - словарь, targets[путь до картинки] = индекс класса
            df=pd.read_csv(annotations_file)
            self.targets = dict(zip([file for file in df["filename"]], [self.class_to_idx[class_] for class_ in df["class"]]))
            

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int]:
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        img_path=self.samples[index]
        image=np.array(Image.open(os.path.join(self.root,img_path)).convert("RGB"))
        if self.transform is not None:
            image=self.transform(image=image)["image"]
        if self.targets is not None and img_path in self.targets:
            return image, img_path, self.targets[img_path]
        return image, img_path, -1


    def __len__(self) -> int:
        """
        Возвращает размер датасета (количество сэмплов).
        """
        return len(self.samples)

def get_resnet(internal_features, transfer=True):
    weights=torchvision.models.ResNet50_Weights.DEFAULT if transfer else None
    model=torchvision.models.resnet50(weights=weights)
    in_feutures=model.fc.in_features
    model.fc=nn.Linear(in_feutures, internal_features)
    return  model
class CustomNetwork(L.LightningModule):
    """
    Класс, реализующий нейросеть для классификации.

    :param features_criterion: loss-функция на признаки, извлекаемые нейросетью перед классификацией (None когда нет такого лосса)
    :param internal_features: внутреннее число признаков
    """

    def __init__(
        self,
        features_criterion: (
            typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None,
        internal_features: int = 1024,
    ):
        super().__init__()
        ### YOUR CODE HERE
        self.lr=LEARNING_RATE
        self.model=get_resnet(internal_features)
        self.linear=nn.Linear(internal_features, CLASSES_CNT)
        self.loss_fn=nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    def accuracy(self, pred, ans):
        assert pred.shape==ans.shape
        return ((pred==ans).sum()/pred.size()[0]).item()

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Функция для прогона данных через нейронную сеть.
        Возвращает два тензора: внутреннее представление и логиты после слоя-классификатора.
        """
        y=self.model(x)
        return y, self.linear(y)

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.

        :param x: батч с картинками
        """
        with torch.no_grad():
            return  np.argmax(self.linear(self.model(x)).cpu().numpy(), axis=-1)
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=GAMMA)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    def training_step(self, batch):
        return self._step(batch, "train")
    def validation_step(self, batch):
        return self._step(batch, "valid")
    def _step(self, batch, kind):
        x, y, _=batch
        p=self.model(x)
        loss=self.loss_fn(p, y)
        accs=self.accuracy(p.argmax(axis=-1), y)
        return self._log_metrics(loss, accs, kind)
    def _log_metrics(self, loss, accs, kind):
        metrics={}
        if loss is not None:
            metrics[f"{kind}_loss"]=loss
        if accs is not None:
            metrics[f"{kind}_accs"]=accs
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )
        return loss
def collate(arr:typing.List[typing.Any]):
    return torch.stack([image for image, _, _ in arr]), torch.tensor([class_id for _, _, class_id in arr]),[img_path for _, img_path,_ in arr]
def train_simple_classifier() -> torch.nn.Module:
    """
    Функция для обучения простого классификатора на исходных данных.
    """
    ### YOUR CODE HERE
    ds_train=DatasetRTSD(root_folders=["cropped-train"], path_to_classes_json="classes.json")
    ds_valid=TestData(root="smalltest", path_to_classes_json="classes.json", annotations_file="smalltest_annotations.csv")
    dl_valid=DataLoader(ds_valid, collate_fn=collate, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    dl_train=DataLoader(ds_train, collate_fn=collate, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    max_epochs=MAX_EPOCHS
    model=CustomNetwork()
    callbacks = [
        L.pytorch.callbacks.TQDMProgressBar(leave=True),
    ]
    trainer = L.Trainer(
        callbacks=callbacks,
        max_epochs=max_epochs,
        enable_checkpointing=False,
        logger=True,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, dl_train, dl_valid)
    torch.save(model.state_dict(), "simple_model.pth")
    return model



def apply_classifier(
    model: torch.nn.Module,
    test_folder: str,
    path_to_classes_json: str,
) -> typing.List[typing.Mapping[str, typing.Any]]:
    """
    Функция, которая применяет модель и получает её предсказания.

    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param path_to_classes_json: путь до файла с информацией о классах classes.json
    """
    ### YOUR CODE HERE - список словарей вида {'filename': 'имя файла', 'class': 'строка-название класса'}
    # state_dict = torch.load(model_path, map_location=config.DEVICE)
    # model.load_state_dict(state_dict)
    ds_test=TestData(test_folder, path_to_classes_json)
    dl_test=DataLoader(ds_test, collate_fn=collate, batch_size=1)
    results = []
    for img, _,img_path in dl_test:
        results.append({'filename':img_path[0], 'class':ds_test.classes[model.predict(img).argmax(axis=-1).item()]})
    return results


def test_classifier(
    model: torch.nn.Module,
    test_folder: str,
    annotations_file: str,
) -> typing.Tuple[float, float, float]:
    """
    Функция для тестирования качества модели.
    Возвращает точность на всех знаках, Recall на редких знаках и Recall на частых знаках.

    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    ### YOUR CODE HERE
    total_acc, rare_recall, freq_recall=0,0,0,0
    ds_test=TestData(test_folder, "classes.json", annotations_file)
    rare_gt=ds_test.rare_gt
    freq_gt=ds_test.freq_gt
    for image, _, target in ds_test:
        pred=model.predict(image).argmax(axis=-1).item()
        total_acc+=(pred==target)
        rare_recall+=(pred in rare_gt)
        freq_recall+=(pred in freq_gt)
    total_acc/=(len(rare_gt)+len(freq_gt))
    rare_recall/=len(rare_gt)
    freq_recall/=len(freq_gt)
    return total_acc, rare_recall, freq_recall


class SignGenerator(object):
    """
    Класс для генерации синтетических данных.

    :param background_path: путь до папки с изображениями фона
    """

    def __init__(self, background_path: str) -> None:
        super().__init__()
        ### YOUR CODE HERE

    ### Для каждого из необходимых преобразований над иконками/картинками,
    ### напишите вспомогательную функцию приблизительно следующего вида:
    ###
    ### @staticmethod
    ### def discombobulate_icon(icon: np.ndarray) -> np.ndarray:
    ###     ### YOUR CODE HERE
    ###     return ...
    ###
    ### Постарайтесь не использовать готовые библиотечные функции для
    ### аугментаций и преобразования картинок, а реализовать их
    ### "из первых принципов" на numpy

    def get_sample(self, icon: np.ndarray) -> np.ndarray:
        """
        Функция, встраивающая иконку на случайное изображение фона.

        :param icon: Массив с изображением иконки
        """
        ### YOUR CODE HERE
        icon = ...
        ### YOUR CODE HERE - случайное изображение фона
        bg = ...
        return  ### YOUR CODE HERE


def generate_one_icon(args: typing.Tuple[str, str, str, int]) -> None:
    """
    Функция, генерирующая синтетические данные для одного класса.

    :param args: Это список параметров: [путь до файла с иконкой, путь до выходной папки, путь до папки с фонами, число примеров каждого класса]
    """
    ### YOUR CODE HERE


def generate_all_data(
    output_folder: str,
    icons_path: str,
    background_path: str,
    samples_per_class: int = 1000,
) -> None:
    """
    Функция, генерирующая синтетические данные.
    Эта функция запускает пул параллельно работающих процессов, каждый из которых будет генерировать иконку своего типа.
    Это необходимо, так как процесс генерации очень долгий.
    Каждый процесс работает в функции generate_one_icon.

    :param output_folder: Путь до выходной директории
    :param icons_path: Путь до директории с иконками
    :param background_path: Путь до директории с картинками фона
    :param samples_per_class: Количество примеров каждого класса, которые надо сгенерировать
    """
    shutil.rmtree(output_folder, ignore_errors=True)
    with ProcessPoolExecutor(8) as executor:
        params = [
            [
                os.path.join(icons_path, icon_file),
                output_folder,
                background_path,
                samples_per_class,
            ]
            for icon_file in os.listdir(icons_path)
        ]
        list(tqdm.tqdm(executor.map(generate_one_icon, params)))


def train_synt_classifier() -> torch.nn.Module:
    """
    Функция для обучения простого классификатора на смеси исходных и ситетических данных.
    """
    ### YOUR CODE HERE
    return model


class FeaturesLoss(torch.nn.Module):
    """
    Класс для вычисления loss-функции на признаки предпоследнего слоя нейросети.
    """

    def __init__(self, margin: float) -> None:
        super().__init__()
        ### YOUR CODE HERE

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Функция, вычисляющая loss-функцию на признаки предпоследнего слоя нейросети.

        :param outputs: Признаки с предпоследнего слоя нейросети
        :param labels: Реальные метки объектов
        """
        ### YOUR CODE HERE


class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """
    Класс для семплирования батчей с контролируемым числом классов и примеров каждого класса.

    :param data_source: Это датасет RTSD
    :param elems_per_class: Число элементов каждого класса
    :param classes_per_batch: Количество различных классов в одном батче
    """

    def __init__(
        self,
        data_source: DatasetRTSD,
        elems_per_class: int,
        classes_per_batch: int,
    ) -> None:
        ### YOUR CODE HERE
        pass

    def __iter__(self):
        """
        Функция, которая будет генерировать список индексов элементов в батче.
        """
        ### YOUR CODE HERE

    def __len__(self) -> None:
        """
        Возвращает общее количество батчей.
        """
        ### YOUR CODE HERE


def train_better_model() -> torch.nn.Module:
    """
    Функция для обучения классификатора на смеси исходных и ситетических данных с новым лоссом на признаки.
    """
    ### YOUR CODE HERE
    return model


class ModelWithHead(CustomNetwork):
    """
    Класс, реализующий модель с головой из kNN.

    :param n_neighbors: Количество соседей в методе ближайших соседей
    """

    def __init__(self, n_neighbors: int) -> None:
        super().__init__()
        self.eval()
        ### YOUR CODE HERE

    def load_nn(self, nn_weights_path: str) -> None:
        """
        Функция, загружающая веса обученной нейросети.

        :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
        """
        ### YOUR CODE HERE

    def load_head(self, knn_path: str) -> None:
        """
        Функция, загружающая веса kNN (с помощью pickle).

        :param knn_path: Путь, откуда надо прочитать веса kNN
        """
        ### YOUR CODE HERE

    def save_head(self, knn_path: str) -> None:
        """
        Функция, сохраняющая веса kNN (с помощью pickle).

        :param knn_path: Путь, куда надо сохранить веса kNN
        """
        ### YOUR CODE HERE

    def train_head(self, indexloader: torch.utils.data.DataLoader) -> None:
        """
        Функция, обучающая голову kNN.

        :param indexloader: Загрузчик данных для обучения kNN
        """
        ### YOUR CODE HERE

    def predict(self, imgs: torch.Tensor) -> np.ndarray:
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.

        :param imgs: батч с картинками
        """
        ### YOUR CODE HERE - предсказание нейросетевой модели
        features, model_pred = ...
        features = features / np.linalg.norm(features, axis=1)[:, None]
        ### YOUR CODE HERE - предсказание kNN на features
        knn_pred = ...
        return knn_pred


class IndexSampler(torch.utils.data.sampler.Sampler[int]):
    """
    Класс для семплирования батчей с картинками индекса.

    :param data_source: Это датасет RTSD с синтетическими примерами
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """

    def __init__(self, data_source: DatasetRTSD, examples_per_class: int) -> None:
        ### YOUR CODE HERE
        pass

    def __iter__(self):
        """
        Функция, которая будет генерировать список индексов элементов в батче.
        """
        return  ### YOUR CODE HERE

    def __len__(self) -> int:
        """
        Возвращает общее количество индексов.
        """
        ### YOUR CODE HERE


def train_head(nn_weights_path: str, examples_per_class: int = 20) -> torch.nn.Module:
    """
    Функция для обучения kNN-головы классификатора.

    :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    ### YOUR CODE HERE


if __name__ == "__main__":
    # The following code won't be run in the test system, but you can run it
    # on your local computer with `python -m rare_traffic_sign_solution`.

    # Feel free to put here any code that you used while
    # debugging, training and testing your solution.
    train_simple_classifier()

