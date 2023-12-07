# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
from tqdm import tqdm
import mat73
import cv2
import scipy
import scipy.io as sio
import random

try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

from utils import open_file

DATASETS_CONFIG = {
    "PaviaC": {
        "urls": [
            "http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat",
            "http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat",
        ],
        "img": "Pavia.mat",
        "gt": "Pavia_gt.mat",
    },
    "Salinas": {
        "urls": [
            "http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
            "http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
        ],
        "img": "Salinas_corrected.mat",
        "gt": "Salinas_gt.mat",
    },
    "PaviaU": {
        "urls": [
            "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
            "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
        ],
        "img": "PaviaU.mat",
        "gt": "PaviaU_gt.mat",
    },
    "KSC": {
        "urls": [
            "http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat",
            "http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat",
        ],
        "img": "KSC.mat",
        "gt": "KSC_gt.mat",
    },
    "IndianPines": {
        "urls": [
            "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
            "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
        ],
        "img": "Indian_pines_corrected.mat",
        "gt": "Indian_pines_gt.mat",
    },
    "Botswana": {
        "urls": [
            "http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat",
            "http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat",
        ],
        "img": "Botswana.mat",
        "gt": "Botswana_gt.mat",
    },
}

try:
    from custom_datasets import CUSTOM_DATASETS_CONFIG

    DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
    """Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None

    # if dataset_name not in datasets.keys():
    #     raise ValueError("{} dataset is unknown.".format(dataset_name))

    # dataset = datasets[dataset_name]

    # folder = target_folder + datasets[dataset_name].get("folder", dataset_name + "/")
    folder = target_folder + dataset_name + "/"
    # if dataset.get("download", True):
    #     # Download the dataset if is not present
    #     if not os.path.isdir(folder):
    #         os.makedirs(folder)
    #     for url in datasets[dataset_name]["urls"]:
    #         # download the files
    #         filename = url.split("/")[-1]
    #         if not os.path.exists(folder + filename):
    #             with TqdmUpTo(
    #                 unit="B",
    #                 unit_scale=True,
    #                 miniters=1,
    #                 desc="Downloading {}".format(filename),
    #             ) as t:
    #                 urlretrieve(url, filename=folder + filename, reporthook=t.update_to)
    # elif not os.path.isdir(folder):
    #     print("WARNING: {} is not downloadable.".format(dataset_name))

    if dataset_name == "PaviaC":
        # Load the image
        img = open_file(folder + "Pavia.mat")["pavia"]

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + "Pavia_gt.mat")["pavia_gt"]

        label_values = [
            "Undefined",
            "Water",
            "Trees",
            "Asphalt",
            "Self-Blocking Bricks",
            "Bitumen",
            "Tiles",
            "Shadows",
            "Meadows",
            "Bare Soil",
        ]

        ignored_labels = [0]

    elif dataset_name == "PaviaU":
        # Load the image
        img = open_file(folder + "PaviaU.mat")["paviaU"]

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + "PaviaU_gt.mat")["paviaU_gt"]

        label_values = [
            "Undefined",
            "Asphalt",
            "Meadows",
            "Gravel",
            "Trees",
            "Painted metal sheets",
            "Bare Soil",
            "Bitumen",
            "Self-Blocking Bricks",
            "Shadows",
        ]

        ignored_labels = [0]

    elif dataset_name == "Salinas":
        img = open_file(folder + "Salinas_corrected.mat")["salinas_corrected"]

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + "Salinas_gt.mat")["salinas_gt"]

        label_values = [
            "Undefined",
            "Brocoli_green_weeds_1",
            "Brocoli_green_weeds_2",
            "Fallow",
            "Fallow_rough_plow",
            "Fallow_smooth",
            "Stubble",
            "Celery",
            "Grapes_untrained",
            "Soil_vinyard_develop",
            "Corn_senesced_green_weeds",
            "Lettuce_romaine_4wk",
            "Lettuce_romaine_5wk",
            "Lettuce_romaine_6wk",
            "Lettuce_romaine_7wk",
            "Vinyard_untrained",
            "Vinyard_vertical_trellis",
        ]

        ignored_labels = [0]

    elif dataset_name == "IndianPines":
        # Load the image
        img = open_file(folder + "Indian_pines_corrected.mat")
        img = img["indian_pines_corrected"]

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + "Indian_pines_gt.mat")["indian_pines_gt"]
        label_values = [
            "Undefined",
            "Alfalfa",
            "Corn-notill",
            "Corn-mintill",
            "Corn",
            "Grass-pasture",
            "Grass-trees",
            "Grass-pasture-mowed",
            "Hay-windrowed",
            "Oats",
            "Soybean-notill",
            "Soybean-mintill",
            "Soybean-clean",
            "Wheat",
            "Woods",
            "Buildings-Grass-Trees-Drives",
            "Stone-Steel-Towers",
        ]

        ignored_labels = [0]

    elif dataset_name == "Botswana":
        # Load the image
        img = open_file(folder + "Botswana.mat")["Botswana"]

        rgb_bands = (75, 33, 15)

        gt = open_file(folder + "Botswana_gt.mat")["Botswana_gt"]
        label_values = [
            "Undefined",
            "Water",
            "Hippo grass",
            "Floodplain grasses 1",
            "Floodplain grasses 2",
            "Reeds",
            "Riparian",
            "Firescar",
            "Island interior",
            "Acacia woodlands",
            "Acacia shrublands",
            "Acacia grasslands",
            "Short mopane",
            "Mixed mopane",
            "Exposed soils",
        ]

        ignored_labels = [0]

    elif dataset_name == "KSC":
        # Load the image
        img = open_file(folder + "KSC.mat")["KSC"]

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + "KSC_gt.mat")["KSC_gt"]
        label_values = [
            "Undefined",
            "Scrub",
            "Willow swamp",
            "Cabbage palm hammock",
            "Cabbage palm/oak hammock",
            "Slash pine",
            "Oak/broadleaf hammock",
            "Hardwood swamp",
            "Graminoid marsh",
            "Spartina marsh",
            "Cattail marsh",
            "Salt marsh",
            "Mud flats",
            "Wate",
        ]

        ignored_labels = [0]
    #####add new code for HHK_Data#######
    elif dataset_name == "HHK":
        # Load the image
        img = open_file(folder + "HHK.mat")["HHK"]

        # rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + "HHK_gt.mat")["HHK_gt"]
        # label_values = [
        #     "Undefined",
        #     "Scrub",
        #     "Willow swamp",
        #     "Cabbage palm hammock",
        #     "Cabbage palm/oak hammock",
        #     "Slash pine",
        #     "Oak/broadleaf hammock",
        #     "Hardwood swamp",
        #     "Graminoid marsh",
        #     "Spartina marsh",
        #     "Cattail marsh",
        #     "Salt marsh",
        #     "Mud flats",
        #     "Wate",
        # ]

        ignored_labels = [0]
    elif dataset_name == 'hhk':
        img1 = mat73.loadmat(folder + 'hhk64.mat')
        img_h = img1['HSI']
        shape_h = img_h.shape
        img_h = (img_h - np.min(img_h)) / (np.max(img_h) - np.min(img_h))
        img_v = img1['VIS']
        # img_v = (img_v - np.min(img_v)) / (np.max(img_v) - np.min(img_v))
        # img_v = cv2.resize(img_v,(shape_h[1],shape_h[0]))
        # img = np.concatenate((img_h, img_v), axis=2)
        img = img_h
        rgb_bands = (108, 68, 32)
        gt = dict()
        gt['Tr'] = img1['Tr']
        gt['Te'] = img1['Te']
        label_values = [
            "Undefined",
            "Scrub",
            "Willow swamp",
            "Cabbage palm hammock",
            "Cabbage palm/oak hammock",
            "Slash pine",
            "Oak/broadleaf hammock",
            "Hardwood swamp",
            "Graminoid marsh",
            "Spartina marsh",
            "Cattail marsh",
            "Salt marsh",
            "Mud flats",
            "Wate",
            "a",
            "b",
            "d"
        ]
        ignored_labels = [0]
    elif dataset_name == 'hhk1728':
        img1 = mat73.loadmat(folder + 'hhk1728.mat')
        img_h = img1['HSI']
        shape_h = img_h.shape
        img_h = (img_h - np.min(img_h)) / (np.max(img_h) - np.min(img_h))
        # img_v = img1['VIS']
        # img_v = (img_v - np.min(img_v)) / (np.max(img_v) - np.min(img_v))
        # img_v = cv2.resize(img_v, (shape_h[1], shape_h[0]))
        # img = np.concatenate((img_h, img_v), axis=2)
        img = img_h
        rgb_bands = (108, 68, 32)
        gt = dict()
        gt['Tr'] = img1['Tr']
        gt['Te'] = img1['Te']
        label_values = [
            "Undefined",
            "Scrub",
            "Willow swamp",
            "Cabbage palm hammock",
            "Cabbage palm/oak hammock",
            "Slash pine",
            "Oak/broadleaf hammock",
            "Hardwood swamp",
            "Graminoid marsh",
            "Spartina marsh",
            "Cattail marsh",
            "Salt marsh",
            "Mud flats",
            "Wate",
            "a",
            "b",
            "d"
        ]
        ignored_labels = [0]
    elif dataset_name == 'hhk880':
        img1 = mat73.loadmat(folder + 'hhk880.mat')
        img_h = img1['HSI']
        shape_h = img_h.shape
        img_h = (img_h - np.min(img_h)) / (np.max(img_h) - np.min(img_h))
        # img_v = img1['VIS']
        # img_v = (img_v - np.min(img_v)) / (np.max(img_v) - np.min(img_v))
        # img_v = cv2.resize(img_v, (shape_h[1], shape_h[0]))
        # img = np.concatenate((img_h, img_v), axis=2)
        img = img_h

        rgb_bands = (108, 68, 32)
        gt = dict()
        gt['Tr'] = img1['Tr']
        gt['Te'] = img1['Te']
        label_values = [
            "Undefined",
            "Scrub",
            "Willow swamp",
            "Cabbage palm hammock",
            "Cabbage palm/oak hammock",
            "Slash pine",
            "Oak/broadleaf hammock",
            "Hardwood swamp",
            "Graminoid marsh",
            "Spartina marsh",
            "Cattail marsh",
            "Salt marsh",
            "Mud flats",
            "Wate",
            "a",
            "b",
            "d"
        ]
        ignored_labels = [0]
    elif dataset_name == 'oil':
        img1 = scipy.io.loadmat(folder + '01.mat')
        img_h = img1['img']
        shape_h = img_h.shape
        img_h = (img_h - np.min(img_h)) / (np.max(img_h) - np.min(img_h))
        # img_v = img1['VIS']
        # img_v = (img_v - np.min(img_v)) / (np.max(img_v) - np.min(img_v))
        # img_v = cv2.resize(img_v, (shape_h[1], shape_h[0]))
        # img = np.concatenate((img_h, img_v), axis=2)
        img = img_h

        rgb_bands = (108, 68, 32)
        gt = img1['map']
        # gt = dict()
        # gt['Tr'] = img1['Tr']
        # gt['Te'] = img1['Te']
        label_values = [
            "Undefined",
            "Scrub",
            "Willow swamp",
            "Cabbage palm hammock",
            "Cabbage palm/oak hammock",
            "Slash pine",
            "Oak/broadleaf hammock",
            "Hardwood swamp",
            "Graminoid marsh",
            "Spartina marsh",
            "Cattail marsh",
            "Salt marsh",
            "Mud flats",
            "Wate",
            "a",
            "b",
            "d"
        ]
        ignored_labels = [0]
    elif dataset_name == 'oil_7':
        img1 = scipy.io.loadmat(folder + '09.mat')
        img_h = img1['img']
        shape_h = img_h.shape
        img_h = (img_h - np.min(img_h)) / (np.max(img_h) - np.min(img_h))
        # img_v = img1['VIS']
        # img_v = (img_v - np.min(img_v)) / (np.max(img_v) - np.min(img_v))
        # img_v = cv2.resize(img_v, (shape_h[1], shape_h[0]))
        # img = np.concatenate((img_h, img_v), axis=2)
        img = img_h

        rgb_bands = (108, 68, 32)
        gt = img1['map']
        # gt = dict()
        # gt['Tr'] = img1['Tr']
        # gt['Te'] = img1['Te']
        label_values = [
            "Undefined",
            "Scrub",
            "Willow swamp",
            "Cabbage palm hammock",
            "Cabbage palm/oak hammock",
            "Slash pine",
            "Oak/broadleaf hammock",
            "Hardwood swamp",
            "Graminoid marsh",
            "Spartina marsh",
            "Cattail marsh",
            "Salt marsh",
            "Mud flats",
            "Wate",
            "a",
            "b",
            "d"
        ]
        ignored_labels = [0]
    elif dataset_name == 'Houston':
        img1 = mat73.loadmat(folder + 'Hu2018.mat')
        img_h = img1['HSI']
        shape_h = img_h.shape
        img_h = (img_h - np.min(img_h)) / (np.max(img_h) - np.min(img_h))
        img_v = img1['LiDAR']
        img_v = (img_v - np.min(img_v)) / (np.max(img_v) - np.min(img_v))
        # img_v = cv2.resize(img_v,(shape_h[1],shape_h[0]))
        # img_v = np.expand_dims(img_v, axis=2)
        img = np.concatenate((img_h, img_v), axis=2)
        # img = img_h
        rgb_bands = (40, 18, 8)
        gt = dict()
        gt['Tr'] = img1['Tr']
        gt['Te'] = img1['Te']
        label_values = [
            "Undefined",
            "Scrub",
            "Willow swamp",
            "Cabbage palm hammock",
            "Cabbage palm/oak hammock",
            "Slash pine",
            "Oak/broadleaf hammock",
            "Hardwood swamp",
            "Graminoid marsh",
            "Spartina marsh",
            "Cattail marsh",
            "Salt marsh",
            "Mud flats",
            "Wate",
            "a",
            "b",
            "d",
            "e",
            "f",
            'g',
            'h',
        ]
        ignored_labels = [0]
    elif dataset_name == 'wz':
        img1 = mat73.loadmat(folder + 'R.mat')
        img_h = img1['R']
        shape_h = img_h.shape
        img_h = (img_h - np.min(img_h)) / (np.max(img_h) - np.min(img_h))
        # img_v = img1['VIS']
        # img_v = (img_v - np.min(img_v)) / (np.max(img_v) - np.min(img_v))
        # img_v = cv2.resize(img_v, (shape_h[1], shape_h[0]))
        # img = np.concatenate((img_h, img_v), axis=2)
        img = img_h

        rgb_bands = (120, 72, 36)
        gt = dict()

        gt['Tr'] = sio.loadmat(folder + 'label.mat')['label']
        gt['Te'] = gt['Tr']
        label_values = [
            "Undefined",
            "Scrub",
            "Willow swamp",
            "Cabbage palm hammock",
            "Cabbage palm/oak hammock",
            "Slash pine",
            "Oak/broadleaf hammock",
            "Hardwood swamp",
            "Graminoid marsh",
            "Spartina marsh",
            "Cattail marsh",
            "Salt marsh",
            "Mud flats",
            "Wate",
            "a",
            "b",
            "d"
        ]
        ignored_labels = [0]
    else:
        # Custom dataset
        (
            img,
            gt,
            rgb_bands,
            ignored_labels,
            label_values,
            palette,
        ) = CUSTOM_DATASETS_CONFIG[dataset_name]["loader"](folder)

    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print(
            "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled."
        )
    img[nan_mask] = 0
    # gt['Tr'][nan_mask] = 0
    # gt['Te'][nan_mask] = 0
    # gt[nan_mask] = 0
    # ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))
    # gt[gt == 0] = 2
    # Normalization
    img = np.asarray(img, dtype="float32")
    # img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img, gt, label_values, ignored_labels, rgb_bands, palette


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.name = 'oil'
        self.patch_size = 11
        self.ignored_labels = set([0])
        self.flip_augmentation = True
        self.radiation_augmentation = True
        self.mixture_augmentation = True
        self.center_pixel = True
        supervision = 'full'
        # Fully supervised : use all pixels with label not ignored
        if supervision == "full":
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == "semi":
            mask = np.ones_like(gt)
        mask[mask == 0] = 2
        mask[int(np.floor(mask.shape[0] / 2)):, int(np.floor(mask.shape[1] / 2)):] = 0
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array(
            [
                (x, y)
                for x, y in zip(x_pos, y_pos)
                if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p
            ]
        )
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1.0, size=2)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert self.labels[l_indice] == value
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]
        data_k = data

        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
            data_k, label1 = self.flip(data_k, label)
        if self.radiation_augmentation and np.random.random() < 0.1:
            data = self.radiation_noise(data)
            data_k = self.radiation_noise(data_k)
        if self.mixture_augmentation and np.random.random() < 0.2:
            data = self.mixture_noise(data, label)
            data_k = self.mixture_noise(data_k, label1)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
        data_k = np.asarray(np.copy(data_k).transpose((2, 0, 1)), dtype="float32")
        label = np.asarray(np.copy(label), dtype="int64")

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        data_k = torch.from_numpy(data_k)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            data_k = data_k[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)
            data_k = data_k.unsqueeze(0)
        # return [data,data_k], label
        return [data, data_k], label


class HyperX_test(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX_test, self).__init__()
        self.data = data
        self.label = gt
        self.name = 'oil'
        self.patch_size = 11
        self.ignored_labels = set([0])
        self.flip_augmentation = True
        self.radiation_augmentation = True
        self.mixture_augmentation = True
        self.center_pixel = True
        supervision = 'full'
        # Fully supervised : use all pixels with label not ignored
        if supervision == "full":
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == "semi":
            mask = np.ones_like(gt)
        # mask[mask == 0] = 2
        x_pos, y_pos = np.nonzero(mask)
        # ind_x = np.where(x_pos % self.patch_size!= 0)
        # ind_y = np.where(y_pos % self.patch_size!= 0)
        # x_pos = np.delete(x_pos, ind_x[0])
        # y_pos = np.delete(y_pos, ind_y[0])
        p = self.patch_size // 2
        self.indices = np.array(
            [
                (x, y)
                for x, y in zip(x_pos, y_pos)
                if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p
            ]
        )
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1.0, size=2)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert self.labels[l_indice] == value
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def four_rotation(self, matrix_0):
        matrix_90 = np.rot90(matrix_0, k=1, axes=(0, 1))
        matrix_180 = np.rot90(matrix_90, k=1, axes=(0, 1))
        matrix_270 = np.rot90(matrix_180, k=1, axes=(0, 1))
        return [matrix_0, matrix_90, matrix_180, matrix_270]

    def rotation(self, matrix_x, matrix_y, pseudo_label=None, segments=None, Mirror=False):
        train_PL, train_SG = [], []
        if pseudo_label is None: train_PL = None
        if segments is None: train_SG = None
        if Mirror == True:
            train_IMG, train_Y = self.four_rotation(matrix_x[::-1, :, :]), self.four_rotation(matrix_y[::-1, :])
            if pseudo_label is not None:
                for k_pseudo_label in pseudo_label:
                    train_PL.append(self.four_rotation(k_pseudo_label[::-1, :, :]))
            if segments is not None:
                for k_segments in segments:
                    train_SG.append(self.four_rotation(k_segments[::-1, :, :]))
        else:
            train_IMG, train_Y = self.four_rotation(matrix_x), self.four_rotation(matrix_y)
            if pseudo_label is not None:
                for k_pseudo_label in pseudo_label:
                    train_PL.append(self.four_rotation(k_pseudo_label))
            if segments is not None:
                for k_segments in segments:
                    train_SG.append(self.four_rotation(k_segments))

        return train_IMG, train_Y, train_PL, train_SG

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]
        data_k = data

        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
            data_k, label1 = self.flip(data_k, label)
        if self.radiation_augmentation and np.random.random() < 0.1:
            data = self.radiation_noise(data)
            data_k = self.radiation_noise(data_k)
        if self.mixture_augmentation and np.random.random() < 0.2:
            data = self.mixture_noise(data, label)
            data_k = self.mixture_noise(data_k, label1)

        train_IMG, train_Y, train_PL, train_SG = self.rotation(data, label)
        train_IMG_M, train_Y_M, train_PL_M, train_SG_M = self.rotation(data, label, Mirror=True)
        rand_id = random.random()
        if rand_id < 0.125:
            data = train_IMG[0]
        elif 0.125 < rand_id < 0.25:
            data = train_IMG[2]
        elif 0.25 < rand_id < 0.375:
            data = train_IMG_M[0]
        elif 0.375 < rand_id < 0.5:
            data = train_IMG_M[2]
        elif 0.5 < rand_id < 0.625:
            data = train_IMG[1]
        elif 0.625 < rand_id < 0.75:
            data = train_IMG_M[1]
        elif 0.75 < rand_id < 0.875:
            data = train_IMG_M[3]
        else:
            data = train_IMG[3]

        train_IMG, train_Y, train_PL, train_SG = self.rotation(data_k, label)
        train_IMG_M, train_Y_M, train_PL_M, train_SG_M = self.rotation(data_k, label, Mirror=True)
        rand_id1 = random.random()
        if rand_id1 < 0.125:
            data_k = train_IMG[0]
        elif 0.125 < rand_id1 < 0.25:
            data_k = train_IMG[2]
        elif 0.25 < rand_id1 < 0.375:
            data_k = train_IMG_M[0]
        elif 0.375 < rand_id1 < 0.5:
            data_k = train_IMG_M[2]
        elif 0.5 < rand_id1 < 0.625:
            data_k = train_IMG[1]
        elif 0.625 < rand_id1 < 0.75:
            data_k = train_IMG_M[1]
        elif 0.75 < rand_id1 < 0.875:
            data_k = train_IMG_M[3]
        else:
            data_k = train_IMG[3]

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
        data_k = np.asarray(np.copy(data_k).transpose((2, 0, 1)), dtype="float32")
        label = np.asarray(np.copy(label), dtype="int64")

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        data_k = torch.from_numpy(data_k)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            data_k = data_k[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)
            data_k = data_k.unsqueeze(0)
        return data, label
        # return [data, data_k], label
