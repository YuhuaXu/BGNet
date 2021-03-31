# -*- coding: UTF-8 -*-
from .kitti_dataset import KITTIDataset
from .kitti12_dataset import KITTI_12_Dataset
# from .mb_dataset import MbDatset
# from .mb_test_dataset import MbTestDatset
__datasets__ = {
    "kitti": KITTIDataset,
    "kitti_12": KITTI_12_Dataset,
    # "mb":MbDatset,
    # "mbtest":MbTestDatset,
}
