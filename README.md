# OrganSegRSTN_PyTorch: an end-to-end coarse-to-fine organ segmentation framework in PyTorch

**This is an re-implementation of OrganSegRSTN in PyTorch 0.4.**

version 0.1 - Jul 31 2018 - by Tianwei Ni, Huangjie Zheng and Lingxi Xie

Original version of OrganSegRSTN is written in CAFFE by Qihang Yu, Yuyin Zhou and Lingxi Xie. Please see https://github.com/198808xc/OrganSegRSTN for more details.

#### If you use our codes, please cite our paper accordingly:

  **Qihang Yu**, Lingxi Xie, Yan Wang, Yuyin Zhou, Elliot K. Fishman, Alan L. Yuille,
    "Recurrent Saliency Transformation Network: Incorporating Multi-Stage Visual Cues for Small Organ Segmentation",
    in IEEE Conference on CVPR, Salt Lake City, Utah, USA, 2018.

https://arxiv.org/abs/1709.04518

###### and possibly, our previous work (the basis of this work):

  **Yuyin Zhou**, Lingxi Xie, Wei Shen, Yan Wang, Elliot K. Fishman, Alan L. Yuille,
    "A Fixed-Point Model for Pancreas Segmentation in Abdominal CT Scans",
    in International Conference on MICCAI, Quebec City, Quebec, Canada, 2017.

https://arxiv.org/abs/1612.08230

All the materials released in this library can **ONLY** be used for **RESEARCH** purposes.

The authors and their institution (JHU/JHMI) preserve the copyright and all legal rights of these codes.

**Before you start, please note that there is a LAZY MODE,
  which allows you to run the entire framework with ONE click.
  Check the contents before Section 4.3 for details.**

## 0. Differences from [OrganSegRSTN](https://github.com/198808xc/OrganSegRSTN)

Improvements:

- We merge `indiv_training.py`, `joint_training.py` into `training.py`
- We merge all `*.prototxt` to `model.py`
- Our code runs almost **twice faster** than original one in CAFFE. 

TODO list:

- The performance of our PyTorch implementation in NIH Pancreas Dataset is **a little poorer** (84.0% - 84.3%) than original one in CAFFE (84.4% - 84.6%). 
  - We are trying different models and weight initialization for higher performance.
- We will re-implement two faster functions "post-processing" and "DSC_computation" in C for python3.
- `coarse_fusion.py` , `oracle_fusion.py`, `oracle_testing.py` will be released soon.
- The pretrained model for our RSTN in PyTorch and `logs/` will be released soon.

Any other suggestions please feel free to contact us.

## 1. Introduction

OrganSegRSTN is a code package for our paper:

  **Qihang Yu**, Lingxi Xie, Yan Wang, Yuyin Zhou, Elliot K. Fishman, Alan L. Yuille,
    "Recurrent Saliency Transformation Network: Incorporating Multi-Stage Visual Cues for Small Organ Segmentation",
    in IEEE Conference on CVPR, Salt Lake City, Utah, USA, 2018.

OrganSegRSTN is a segmentation framework designed for 3D volumes.
    It was originally designed for segmenting abdominal organs in CT scans,
    but we believe that it can also be used for other purposes,
    such as brain tissue segmentation in fMRI-scanned images.

OrganSegRSTN is based on the state-of-the-art deep learning techniques.
    This code package is to be used with *PyTorch*, a deep learning library.

It is highly recommended to use one or more modern GPUs for computation.
    Using CPUs will take at least 50x more time in computation.

**We provide an easy implementation in which the training stages has only 1 fine-scaled iteration.
  If you hope to add more, please modify the prototxt file accordingly.
  As we said in the paper, our strategy of using 1 stage in training and multiple iterations in testing works very well.**

## 2. File List

| Folder/File                 | Description                                         |
| :-------------------------- | :-------------------------------------------------- |
| `README.md`                 | the README file                                     |
|                             |                                                     |
| **DATA2NPY/**               | codes to transfer the NIH dataset into NPY format   |
| `dicom2npy.py`              | transferring image data (DICOM) into NPY format     |
| `nii2npy.py`                | transferring label data (NII) into NPY format       |
|                             |                                                     |
| **OrganSegRSTN/**           | primary codes of OrganSegRSTN                       |
| `coarse2fine_testing.py`    | the coarse-to-fine testing process                  |
| `coarse_testing.py`         | the coarse-scaled testing process                   |
| `Data.py`                   | the data layer                                      |
| `model.py`                   | the models of RSTN                                     |
| `init.py`                   | the initialization functions                        |
| `training.py`         | training the coarse and fine stages jointly         |
| `run.sh`                    | the main program to be called in bash shell         |
| `utils.py`                  | the common functions                                |
|                             |                                                     |


## 3. Installation


#### 3.1 Prerequisites

###### 3.1.1 Please make sure that your computer is equipped with modern GPUs that support CUDA.

    Without them, you will need 50x more time in both training and testing stages.

###### 3.1.2 Please also make sure that python (we are using 3.6) is installed.

#### 3.2 PyTorch

###### 3.2.1 Download a PyTorch library from https://pytorch.org/ . We are using PyTorch 0.4.0 .


## 4. Usage

Please follow these steps to reproduce our results on the NIH pancreas segmentation dataset.

**NOTE**: Here we only provide basic steps to run our codes on the NIH dataset.
    For more detailed analysis and empirical guidelines for parameter setting
    (this is very important especially when you are using our codes on other datasets),
    please refer to our technical report (check our webpage for updates).


#### 4.1 Data preparation

###### 4.1.1 Download NIH data from https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT .
    You should be able to download image and label data individually.
    Suppose your data directory is $RAW_PATH:
        The image data are organized as $RAW_PATH/DOI/PANCREAS_00XX/A_LONG_CODE/A_LONG_CODE/ .
        The label data are organized as $RAW_PATH/TCIA_pancreas_labels-TIMESTAMP/label00XX.nii.gz .

###### 4.1.2 Use our codes to transfer these data into NPY format.
    Put dicom2npy.py under $RAW_PATH, and run: python dicom2npy.py .
        The transferred data should be put under $RAW_PATH/images/
    Put nii2npy.py under $RAW_PATH, and run: python nii2npy.py .
        The transferred data should be put under $RAW_PATH/labels/

###### 4.1.3 Suppose your directory to store experimental data is `$DATA_PATH`:

    Put images/ under $DATA_PATH/
    Put labels/ under $DATA_PATH/
    Download the FCN8s pretrained model below and put it under $DATA_PATH/models/pretrained/

[The FCN8s pretrained model in PyTorch](https://drive.google.com/uc?id=0B9P1L--7Wd2vT0FtdThWREhjNkU) - see the explanations in 4.2.3.

NOTE: If you use other path(s), please modify the variable(s) in run.sh accordingly.

#### 4.2 Initialization (requires: 4.1)

###### 4.2.1 Check `run.sh` and set $DATA_PATH accordingly.

###### 4.2.2 Set `$ENABLE_INITIALIZATION=1` and run this script.
    Several folders will be created under $DATA_PATH:
        $DATA_PATH/images_X|Y|Z/: the sliced image data (data are sliced for faster I/O).
        $DATA_PATH/labels_X|Y|Z/: the sliced label data (data are sliced for faster I/O).
        $DATA_PATH/lists/: used for storing training, testing and slice lists.
        $DATA_PATH/logs/: used for storing log files during the training process.
        $DATA_PATH/models/: used for storing models (snapshots) during the training process.
        $DATA_PATH/results/: used for storing testing results (volumes and text results).
    According to the I/O speed of your hard drive, the time cost may vary.
        For a typical HDD, around 20 seconds are required for a 512x512x300 volume.
    This process needs to be executed only once.
    
    NOTE: if you are using another dataset which contains multiple targets,
        you can modify the variables "ORGAN_NUMBER" and "ORGAN_ID" in run.sh,
        as well as the "is_organ" function in utils.py to define your mapping function flexibly.


![](icon.png)
**LAZY MODE!**
![](icon.png)

You can run all the following modules with **one** execution!
  * a) Enable everything (except initialization) in the beginning part.
  * b) Set all the "PLANE" variables as "A" (4 in total) in the following part.
  * c) Run this manuscript!


#### 4.3 Training (requires: 4.2)

###### 4.3.1 Check `run.sh` and set `$TRAINING_PLANE` and `$TRAINING_GPU`.

    You need to run X|Y|Z planes individually, so you can use 3 GPUs in parallel.
    You can also set TRAINING_PLANE=A, so that three planes are trained orderly in one GPU.

###### 4.3.2 Set `$ENABLE_TRAINING=1` and run this script.

    The following folders/files will be created:
        Under $DATA_PATH/models/snapshots/, a folder named by training information.
            Snapshots will be stored in this folder.
    On the axial view (training image size is 512x512, small input images make training faster),
        each 20 iterations cost ~10s on a Titan-X Pascal GPU, or ~8s on a Titan-Xp GPU.
        As described in the code, we need ~80K iterations, which take less than 5 GPU-hours.

###### 4.3.3 Important notes on initialization and model convergence.

![](icon.png)
![](icon.png)
![](icon.png)
![](icon.png)
![](icon.png)
![](icon.png)
![](icon.png)
![](icon.png)

It is very important to provide a reasonable initialization for our model.
In the previous step of data preparation, we provide a scratch model for the NIH dataset,
in which both the coarse and fine stages are initialized using the weights of an FCN-8s model
(please refer to the [FCN project](https://github.com/shelhamer/fcn.berkeleyvision.org)).
This model was pre-trained on PASCALVOC.

###### How to determine if a model converges and works well?

The DSC value in the beginning of training is almost 0.0.
If a model converges, you should observe the loss function values to decrease gradually.
**But in order to make it work well, in the last several epochs,
you need to confirm the average DSC value to be sufficiently high (e.g. 0.8 - 0.85).**

###### Training RSTN on other CT datasets?

If you are experimenting on other **CT datasets**, we strongly recommend you to use a pre-trained model,
such as those pre-trained model attached in the last part of this file.
We also provide [a mixed model](http://nothing) (to be provided soon),
which was tuned using all X|Y|Z images of 82 training samples for pancreas segmentation on NIH.
Of course, do not use it to evaluate any NIH data, as all cases have been used for training.

#### 4.4 Coarse-scaled testing (requires: 4.3)

###### 4.4.1 Check `run.sh` and set `$COARSE_TESTING_PLANE` and `$COARSE_TESTING_GPU`.

    You need to run X|Y|Z planes individually, so you can use 3 GPUs in parallel.
    You can also set COARSE_TESTING_PLANE=A, so that three planes are tested orderly in one GPU.

###### 4.4.2 Set `$ENABLE_COARSE_TESTING=1` and run this script.

    The following folder will be created:
        Under $DATA_PATH/results/, a folder named by training information.
    Testing each volume costs ~30 seconds on a Titan-X Pascal GPU, or ~25s on a Titan-Xp GPU.

#### 4.5 Coarse-to-fine testing (requires: 4.4)

###### 4.5.1 Check run.sh and set `$COARSE2FINE_TESTING_GPU`.

    Fusion is performed on CPU and all X|Y|Z planes are combined.
    Currently X|Y|Z testing processes are executed with one GPU, but it is not time-comsuming.

###### 4.5.2 Set `$ENABLE_COARSE2FINE_TESTING=1` and run this script.

    The following folder will be created:
        Under $DATA_PATH/results/, a folder named by coarse-to-fine information (very long).
    This function calls both fine-scaled testing and fusion codes, so both GPU and CPU are used.
        In our future release, we will implement post-processing in C for acceleration.

**NOTE**: currently we set the maximal rounds of iteration to be 10 in order to observe the convergence.
    Most often, it reaches an inter-DSC of >99% after 3-5 iterations.
    If you hope to save time, you can slight modify the codes in coarse2fine_testing.py.
    Testing each volume costs ~40 seconds on a Titan-X Pascal GPU, or ~32s on a Titan-Xp GPU.
    If you set the threshold to be 99%, this stage will be done within 2 minutes (in average).


Congratulations! You have finished the entire process. Check your results now!

## 5. Versions

The current version is v0.1.

## 6. Contact Information

If you encounter any problems in using these codes, please open an issue in this repository.
You may also contact Tianwei Ni (twni2016@gmail.com) or Lingxi Xie (198808xc@gmail.com).

Thanks for your interest! Have fun!

