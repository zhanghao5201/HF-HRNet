# HF-HRNet
About the repo for paper: HF-HRNet: A Simple Hardware Friendly High-Resolution Network.

Paper is available at [HF-HRNet](https://ieeexplore.ieee.org/abstract/document/10472506/)

# **News**
2023/09/18 Code is open source.

2024/03/05 Paper is accepted as a Transactions Paper for publication with no further changes in an upcoming issue of the IEEE Transactions on Circuits and Systems for Video Technology.


# **Abstract**:

High-resolution networks have made significant progress in dense prediction tasks such as human pose estimation and semantic segmentation. To better explore this high-resolution mechanism on mobile devices, Lite-HRNet incorporates shuffle operations to reduce computational complexity in the channel dimension, while Dite-HRNet employs dynamic convolution and pooling to capture long-range interactions with low computational complexity in the spatial dimension. 
The core idea behind both approaches is to efficiently capture information in either the channel or spatial dimension.
However, shuffle operations and dynamic operations are not hardware-friendly. As a result, both Lite-HRNet and Dite-HRNet cannot achieve the desired inference speed on specialized devices, including Neural Processing Units (NPUs) and Graphics Processing Units (GPUs).
To overcome these limitations, we present a simple Hardware-Friendly Lightweight High-resolution Network (HF-HRNet) based on our proposed Hardware-Friendly Uniform-sized Mug (HUM) block. HUM block mainly consists of the Cascaded Depthwise (CAD) block and Multi-Scale Context Embedding (MCE) block. The CAD block cascades depthwise convolutions to obtain a larger receptive field in the spatial dimension, while the MCE block aggregates multi-scale spatial feature information from different scales and adjusts channel features.
Extensive experiments are conducted on human pose estimation (COCO, MPII) and semantic segmentation (Cityscapes), resulting in a better trade-off between inference speed and accuracy on both NPUs and GPUs.
It is noteworthy that on the COCO test-dev set, HF-HRNet-30 outperforms Dite-HRNet-30 and Lite-HRNet-30 by 1.9 AP and 2.8 AP, respectively, while running about 13 times faster and 9 times faster on NPUs, respectively.

### Prepare datasets

It is recommended to symlink the dataset root to `$HF_HRNET/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. [HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation) provides person detection result of COCO val2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-)
Download and extract them under `$HF_HRNET/data`, and make them look like this:

```
HF-HRNet
├── configs
├── tools
`── data
    │── coco
        │-- annotations
        │   │-- person_keypoints_train2017.json
        │   |-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        │-- train2017
        │   │-- 000000000009.jpg
        │   │-- 000000000025.jpg
        │   │-- 000000000030.jpg
        │   │-- ...
        `-- val2017
            │-- 000000000139.jpg
            │-- 000000000285.jpg
            │-- 000000000632.jpg
            │-- ...

```

**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/).
We have converted the original annotation files into json format, please download them from [mpii_annotations](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/datasets/mpii_annotations.tar).
Extract them under `$HF_HRNET/data`, and make them look like this:

```
HF-HRNet
├── configs
├── tools
`── data
    │── mpii
        |── annotations
        |   |── mpii_gt_val.mat
        |   |── mpii_test.json
        |   |── mpii_train.json
        |   |── mpii_trainval.json
        |   `── mpii_val.json
        `── images
            |── 000001163.jpg
            |── 000003072.jpg

```


# **Acknowledgement**:
This project is developed based on the [MMPOSE](https://github.com/open-mmlab/mmpose)

# **Citation**:

If our code or models help your work, please cite HF-HRNet (TCSVT):
```BibTeX
@article{zhang2024hf,
  title={HF-HRNet: A Simple Hardware Friendly High-Resolution Network},
  author={Zhang, Hao and Dun, Yujie and Pei, Yixuan and Lai, Shenqi and Liu, Chengxu and Zhang, Kaipeng and Qian, Xueming},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  doi={10.1109/TCSVT.2024.3377365},
  publisher={IEEE}
}
```

