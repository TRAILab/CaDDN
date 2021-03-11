<img src="docs/trailab.png" align="right" width="30%">

# CaDDN

`CaDDN` is a monocular-based 3D object detection method. This repository is heavily based off of [`[OpenPCDet]`](https://github.com/open-mmlab/OpenPCDet).

**Categorical Depth Distribution Network for Monocular 3D Object Detection**\
Cody Reading, Ali Harakeh, Julia Chae, and Steven L. Waslander\
**[[Paper](https://arxiv.org/abs/2103.01100)]**


## Overview
- [Changelog](#changelog)
- [Model Zoo](#model-zoo)
- [Installation](docs/INSTALL.md)
- [Getting Started](docs/GETTING_STARTED.md)
- [Citation](#citation)


## Introduction
[2021-03-16] `CaDDN` v0.1.0 is released.

## Model Zoo

### KITTI 3D Object Detection Baselines
Selected supported methods are shown in the below table. The results are the 3D detection performance of moderate difficulty on the *val* set of KITTI dataset.
* All models are trained with 8 GTX 1080Ti GPUs and are available for download.
* The training time is measured with 8 TITAN XP GPUs and PyTorch 1.5.

|                                             | training time | Car@R11 | Pedestrian@R11 | Cyclist@R11  | download |
|---------------------------------------------|----------:|:-------:|:-------:|:-------:|:---------:|
| [PointPillar](tools/cfgs/kitti_models/pointpillar.yaml) |~1.2 hours| 77.28 | 52.29 | 62.68 | [model-18M](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing) |

% TO DO: Change metrics to CaDDN

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of `CaDDN`.

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.


## License

`CaDDN` is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
`CaDDN` is an open source project for monocular-based 3D scene perception.
We would like to thank the authors of `OpenPCDet` for their open-source release of their 3D object detection codebase.


## Citation
If you find this project useful in your research, please consider cite:
```
@article{CaDDN,
    title={Categorical Depth DistributionNetwork for Monocular 3D Object Detection},
    author={Cody Reading and
            Ali Harakeh and
            Julia Chae and
            Steven L. Waslander},
    journal = {CVPR},
    year={2021}
}
```


## Contribution
Welcome to be a member of the CaDDN development team by contributing to this repo, and feel free to contact us for any potential contributions.


