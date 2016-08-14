
## Image Conditioned Noise Model (ICNM)
Created by [Ishan Misra](http://www.cs.cmu.edu/~imisra/)

Based on the CVPR 2016 Paper - "Seeing through the Human Reporting Bias : Visual Classifiers from Noisy Human-Centric Labels"

You can download the paper [here](http://arxiv.org/abs/1512.06974).

### Why use it?
This codebase can help you replicate the results from the paper and train new models.
In cases where your image labels are noisy, our method provides *orthogonal gains over existing techniques*. It may help boost performance of your baseline classification system.
This paper introduces a way to learn visually correct classifiers from noisy labels.
It is an Image Conditioned Noise Model which can estimate label noise *without clean labels*.

### Introduction

Our code base is a mix of Python and C++ and uses the [Caffe](https://github.com/BVLC/caffe) framework.

It is heavily derived from the [visual concepts codebase](https://github.com/s-gupta/visual-concepts) by Saurabh Gupta, and the [Fast-RCNN codebase](https://github.com/rbgirshick/fast-rcnn) by Ross Girshick.
It also uses the [MS COCO PythonAPI](https://github.com/pdollar/coco/tree/master/PythonAPI) from Piotr Dollar.

### Citing

If you find our code useful in your research, please consider citing:
```
   @inproceedings{MisraNoisy16,
    Author = {Ishan Misra and C. Lawrence Zitnick and Margaret Mitchell and Ross Girshick},
    Booktitle = {CVPR},
    Title = {{Seeing through the Human Reporting Bias:  Visual Classifiers from Noisy Human-Centric Labels}},
    Year = {2016},
   }
```

### Contents
1. [Requirements: software](#requirements-software)
2. [Installation](#installation)
3. [Download ground truth](#download-ground-truth)
4. [Usage](#usage)
5. [Training and Experiment scripts](#training-and-experiment-scripts)
6. [Extra downloads](#extra-downloads)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers and OpenCV.

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  USE_OPENCV := 1
  ```

2. Python packages you might not have: `python-opencv`, `nltk`. You can skip `nltk` if you download pre-computed ground-truth files (links below).

### Installation

1. Clone the ICNM repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/imisra/icnm.git
  ```

2. We'll call the directory that you cloned ICNM into `ICNM_ROOT`
The following subdirectories should exist as soon as you clone
- `caffe-ICNM` : contains the caffe version used by this codebase
- `utils` : utilities for loading/saving data, reading caffe logs, and simple MAP/REDUCE jobs
- `vocabs` : vocabulary files (classes) for visual concepts
- `coco` : PythonAPI for MS COCO dataset
- `experiments` : prototxt (solver, train, deploy) files for the models
  + `baselines` : baseline models [only prototxt]
  + `latentNoise` : models that use our method [only prototxt]

3. Build Caffe and pycaffe
    ```Shell
    cd $ICNM_ROOT/caffe-icnm
    # Now follow the Caffe installation instructions here:
    # http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 pycaffe #makes caffe and pycaffe with 8 processes in parallel
    ```

4. Download pre-computed ICNM classifiers [here](https://goo.gl/gf8IjP) and unzip.
  This will populate the folder `$ICNM_ROOT/experiments/latentNoise/cache` with caffe model files.

5. Download COCO Dataset and annotations
    ```Shell
   	wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
    wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
    wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip
	 ```

6. Extract all of these zips into one directory named `$ICNM_ROOT/data/coco`. You can optionally extract them in another location, say $COCO_ROOT and create a symlink to that location.

	```Shell
	unzip train2014.zip
    unzip val2014.zip
    unzip captions_train-val2014.zip
	```

7. It should have this basic structure

	```Shell
  	$ICNM_ROOT/data/coco/images               # images
  	$ICNM_ROOT/data/coco/images/train2014     # images
  	$ICNM_ROOT/data/coco/images/val2014       # images
  	$ICNM_ROOT/data/coco/annotations          # json files with annotations
	```

8. [Optional] Download baseline models for COCO [here](https://goo.gl/uxrbuI).
  Unzipping this in `$ICNM_ROOT` will populate the folder `$ICNM_ROOT/experiments/latentNoise/cache` with caffe model files.
### Download ground truth
Use these steps to train and test our model on the COCO dataset.

We bundle up training labels, evaluation labels, and other files used for training/testing our network.
This step makes it so that the `nltk` package is not used. It also includes the `json` files for the 20k test split used in our paper. Following the MILVC paper, we call this split as `valid2`. It is half the size of the official COCO 2014 val set. Download it [here](https://goo.gl/xU2k5V)

This should give you the following files
- `coco1k_coco-valid2_label_counts.h5`: Ground truth for 1000 visual concepts
- `coco_instancesGT_eval_*` : COCO detection ground truth converted to classification ground truth
- `labels_captions_coco_vocabS1k_train.h5` and `ids_captions_coco_vocabS1k_train.txt` : Label files used to *train* models
- `captions_*.json`: COCO captions ground-truth files for `valid2` split. Place them under the `annotations` directory of your COCO dataset.

### Usage
Once you have models and data downloaded, you can run the `main_test.py` file. This should reproduce the results from the paper by first classifying 20k images from the `valid2` set and then evaluating the result. The file has documentation to understand the various types of evaluation and baseline methods. Since, this process is time consuming, we also provide the pre-computed detections and evaluations in the [Extra downloads](#extra-downloads) section for the baseline and our method.

### Training and Experiment scripts
Scripts to reproduce the experiments in the paper (*up to stochastic variation*) are provided in `$ICNM_ROOT/experiments/scripts`. To train the methods, ensure that the label ground truth files and image paths in the prototxt are correct.
Example, in the `$ICNM_ROOT/experiments/baselines/classification_finetune.prototxt` file, the following paths should exist.
```
mil_data_param {
  label_file: "./data/labels_captions_coco_vocabS1k_train.h5"
  source: "./data/ids_captions_coco_vocabS1k_train.txt"
  root_dir: "./data/coco/images/train2014"
}
```

Log files for experiments can be downloaded from [Experiment logs](). After unzipping they are located in `experiments/baseline/logs` and `experiments/latentNoise/logs`.

*Note*: Our models (in the `experiments/latentNoise` subdirectory), require you to train the baseline (`experiments/baseline`) models first, for two epochs, and are initialized using these models. For the COCO dataset, one epoch is roughly 80k iterations with a batch size of 1. Thus, our models are initialized by baseline models trained for 160k iterations.

*Additional details on the data layer*: The data layer used is from the visual concepts codebase. Briefly, it takes in the following parameters
- `n_classes`: number of classes
- `source`: a list of file containing the image names (without file extension)
- `label_file`: a h5 file with key as the image name (without file extension), and value as a vector of class labels size `1 x n_classes x 1 x 1`
- `root_dir`: root directory containing images. Paths in the `source` file must be relative to this directory.

### Extra downloads
- [Experiment logs](https://goo.gl/yKhf0q)
- ImageNet pre-trained models. These models are fully convolutional.
 - [VGG16 & AlexNet fully-conv](https://goo.gl/9BVvRF) : download size 706MB
- [Classification and Evaluation files](https://goo.gl/VviUSD) : COCO `valid2` set detections for methods in Table 1 from the paper: `MILVC, MILVC + Latent, Classification, Classification + Latent`. Download size is 3.5GB and expands to 7.6GB (poor compression by `h5py` results in big files).
  These files expand into the following directory structure: `$ICNM_ROOT/det-output/baselines/<method>/*` and `$ICNM_ROOT/det-output/latentNoise/<method>/*`. To read and print the data in these files please look at the function `mainTest` in file `main_test.py`.
  It will print out the values in plain text (`cap_eval_utils.print_benchmark_text`) or formatted in latex (`cap_eval_utils.print_benchmark_latex`) according to our Table 1 in our paper.
