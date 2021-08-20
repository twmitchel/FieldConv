# FieldConv

The official implementation of field convolution from the ICCV 2021 paper and oral presentation.

<img src="fig/scatter_vs_gather.png" width="80%">

## [[Paper: Field Convolutions for CNNs on Surfaces]](https://arxiv.org/abs/2104.03916)

Field convolutions are highly discriminating, flexible, and straight-forward to implement. The goal of this repository is to provide the tools for incorperating field convolutions into arbitrary surface learning frameworks. 

## Contents
  - [Dependencies](#dependencies)
  - [Installation](#installation)
  - [Experiments](#experiments)
  - [Authorship and Acknowledgements](#author)

## Dependencies
- [PyTorch >= 1.9](https://pytorch.org)
- [PyTorch Scatter + PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [Progressbar2](https://pypi.org/project/progressbar2/)
- [Suitesparse](http://faculty.cse.tamu.edu/davis/suitesparse.html) (used for the Vector Heat Method)
- [CMake](https://cmake.org/)

Our implementation relies on PyTorch's support for complex numbers introduced in version 1.9. The majority of this code is not compatable with earlier PyTorch versions. PyTorch Geometric is used only for data pre-processing and PyTorch Scatter is used in the convolution modules. Suitesparse + CMake are used for building the fcutils extension used in data pre-processing.

## Installation
Clone this repository and its submodules
```
$ git clone --recurse-submodules https://github.com/twmitchel/FieldConv.git
```
Data pre-processing makes use of the `fcutils` python module, a binding exposing C++ routines from [Geometry Central](https://geometry-central.net) (for computing log maps, transport, etc. on meshes) and a few other routines.

The module is installed with pip by running the following command in the main directory:
```
$ pip install ./fcutils
```

## Experiments
This repository contains four Jupyter Notebooks which can be run to replicate the individual experiments in the paper. If either citing our results on these benchmarks or using/repurosing any of these experiments in your own work, please remember to cite the authors of the original datasets included below.

#### Shape Classification
Create the subdirectory 
```
/data/SHREC11/raw/
```
then download the [SHREC '11 Shape Classification dataset](https://www.cs.jhu.edu/~misha/Code/FieldConvolutions/SHREC11.zip) [1] and place the `SHREC11.zip` file in the above subdirectory. 
Afterwards, run the `classification.ipynb` notebook in the root directory. ([Original dataset](https://www.nist.gov/itl/iad/shrec-2011-shape-retrieval-contest-non-rigid-3d-watertight-meshes))

#### Shape Segmentation
Create the subdirectory
```
/data/SHAPESEG/raw/
```
then download the [Composite Humanoid Shape Segmentation dataset](https://www.cs.jhu.edu/~misha/Code/FieldConvolutions/SHAPESEG.zip) [2] and place the `SHAPESEG.zip` file in the above subdirectory.
Afterwards, run the `segmentation.ipynb` notebook in the root directory. ([Original dataset](https://github.com/Haggaim/ToricCNN))

#### Dense Correspondence
Create the subdirectory
```
/data/FAUSTRM/raw/
```
then download the [remeshed FAUST dataset](https://www.cs.jhu.edu/~misha/Code/FieldConvolutions/FAUSTRM.zip) [3, 4] and place the `FAUSTRM.zip` file in the above subdirectory.
Afterwards, run the `correspondence.ipynb` notebook in the root directory. ([Original dataset](http://faust.is.tue.mpg.de/) | [Remeshed](https://github.com/LIX-shape-analysis/GeomFmaps))

#### Feature Matching
We have contacted the authors of the [Isometric and Non-Isometric Shape Correspondence Benchmark dataset](https://shrec19.cs.cf.ac.uk/) [5] and they plan to make the ground truth correspondences publicly available. We will provide instructions on how to organize and pre-process the dataset so our experiments can be replicated. 

#### References
<small>[1] Zhouhui Lian et al. 2011. SHREC ’11 Track: Shape
Retrieval on Non-rigid 3D Watertight Meshes. Eurographics Workshop on 3D Object
Retrieval.
 
[2] Haggai Maron, Meirav Galun, Noam Aigerman, Miri Trope, Nadav Dym, Ersin Yumer,
Vladimir G Kim, and Yaron Lipman. 2017. Convolutional neural networks on surfaces
via seamless toric covers. ACM Trans. Graph 36, 4 (2017). 
 
[3] Federica Bogo, Javier Romero, Matthew Loper, and Michael J. Black. 2014. FAUST:
Dataset and evaluation for 3D mesh registration. In CVPR. IEEE.
 
[4] Nicolas Donati, Abhishek Sharma , and Maks Ovsjanikov. 2020. Deep geometric functional maps:
Robust feature learning for shape correspondence. In CVPR. IEEE.
 
[5] Roberto M. Dyke et al. 2019. Shape Correspondence with Isometric and Non-Isometric Deformations.
Eurographics Workshop on 3D Object Retrieval.
</small>

<hr/>

## Authorship and Acknowledgements

Author: Thomas (Tommy) Mitchel (tmitchel 'at' jhu 'dot' edu)

Please cite our paper if this code or our method contributes to a publication:
```
@article{mitchel2021field,
  title={Field Convolutions for Surface CNNs},
  author={Mitchel, Thomas W and Kim, Vladimir G and Kazhdan, Michael},
  journal={arXiv preprint arXiv:2104.03916},
  year={2021}
}
```

Much of this code is adapted from the publicly available code for [HSN](https://github.com/rubenwiersma/hsn) (Wiersma et al. 2020) which serves an excellent template for implementing equivariant surface networks. 
