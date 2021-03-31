# BGNet

Codes and models for our paper:

Bin Xu, Yuhua Xu, Xiaoli Yang, Wei Jia, Yulan Guo. Bilateral Grid Learning for Stereo Matching Network. CVPR, 2021.

predict_sample.py is a demo for disparity prediction with two images as input.

Six models are provided:

---kitti_12_BGNet.pth

---kitti_12_BGNet_Plus.pth

---kitti_15_BGNet_Plus.pth

---Sceneflow-BGNet.pth

---Sceneflow-BGNet-Plus.pth

---Sceneflow-IRS-BGNet.pth

For details of the network architectures of BGNet and BGNet+, please see bgnet.py and bgnet_plus.py.

## Installation

---Pytorch 1.3

---CUDA 10.1

---Python 3.6


## Citation

```
@inproceedings{xu2021bilateral,
  title={Bilateral Grid Learning for Stereo Matching Networks},
  author={Xu, Bin and Xu, Yuhua and Yang, Xiaoli and Jia, Wei and Guo, Yulan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1--10},
  year={2021}
}
```
