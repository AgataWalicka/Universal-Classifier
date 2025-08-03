# Universal classifier

This project enables training and validating the performance of the neural network designed for semantic segmentation of point clouds. The code uses [SparseConvNet library](https://github.com/facebookresearch/SparseConvNet) [[1]](#1), [[2]](#2) and Pytorch. The work was performed in the frames of project funded by National Science Centre (Poland) titled: ["Development of a deep learning-based methodology for reproducible classification of airborne laser scanning point clouds with different characteristics"](https://projekty.ncn.gov.pl/en/index.php?projekt_id=519805) (UMO-2021/41/N/ST10/02996).

## Setup
The code was tested on Ubuntu 22.04 using Cuda 11.3 and Python 3.10.12 with the following libraries:
* laspy == 2.1.2
* numpy == 1.22.3
* PyTorch == 1.11.0
* SparseConvNet == 0.2
* tomli == 2.0.1
* matplotlib == 2.5.1

## Examples
The code should be run using cmd console with an argument defining a location of the configuration file in toml format.

Example: Training the model
```commandline
python3 unet.py config.toml
```

Example: Testing the model
```commandline
python3 test_unet.py config.toml
```

For the inference of the already trained model on the data that is not labeled, an artificial classification feature should be added to the point cloud with the value of, for instance, 1.

## References
<a id="1">[1]</a> Graham, B., & Van der Maaten, L. (2017). Submanifold sparse convolutional networks. arXiv preprint arXiv:1706.01307. \
<a id="2">[2]</a> Graham, B., Engelcke, M., & Van Der Maaten, L. (2018). 3d semantic segmentation with submanifold sparse convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 9224-9232).
