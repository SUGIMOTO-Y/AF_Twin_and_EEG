# AF-Twin and AF-EEG
![fig1](overview.png)

This is the Pytorch implementation of this paper: **Cross-subject EEG-based Visual Object Recognition: A Contrastive and Transfer Learning Approach**
	
Yuma Sugimoto<sup>1</sup>, Genci Capi<sup>2</sup>

<sup>1</sup>the Graudate School of Science and Engineering, Hosei University, Koganei, Tokyo, Japan.

<sup>2</sup>the Department of Mechanical Engineering, Hosei University, Koganei, Tokyo, Japan.
***
## Dataset
See Section II, Subsection C, Paragraph 1 for dataset and preprocessing details.

The Things-EEG2 dataset can be downloaded following the official links. They are lireased by Gifford, A. T. in the [A large and rich EEG dataset for modeling human visual object recognition](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub).
* [Thing-EEG2](https://osf.io/3jk45/)

### Dataset Structure Requirement
```
AF_Tiwn_and_EEG
└-- Preprocesssed_ThingEEGDataset
    |-- sub-01
    |	└-- preprocessed_eeg_test.npy
    |	└-- preprocessed_eeg_training.npy
    |-- sub-02
    |	└-- preprocessed_eeg_test.npy
    |	└-- preprocessed_eeg_training.npy
    '
    '
    '
    └-- sub-10
    	└-- preprocessed_eeg_test.npy
    	└-- preprocessed_eeg_training.npy
```
***
## Pretraining Weights
Please download pretrained weights from the following link and unzip them in the project directory.
* [Pretraining_Weights_link](https://drive.google.com/file/d/1ANN6oq5KM1Ei9g3o862SSEbVWgvedrq6/view?usp=sharing) (over 50 GB)

***
## Enviroment
* Ubuntu==20.04.6 LTS
* Python==3.8.10
* CUDA==12.4
### Install
* pytorch==2.2.2
* numpy==1.24.4
* omegaconf==2.3.0
* tqdm==4.66.2

***
## Start Finetuning and Evaluation
```bash
$ python -m Main -s 1 -a 0 -f
```
* '-s' argument means the target subject. (from 1 to 10)
* '-a' argument manages which uses AF-Twin or AF-EEG (0.AF-Twin, 1.AF-EEG).
```bash
$ python -m Main -h
usage: Main.py [-h] [-s SUBJECT] [-a ADAPTER_MODE] [-f FINETUNING]

optional arguments:
  -h, --help            show this help message and exit
  -s SUBJECT, --subject SUBJECT
                        The target subject (from 1 to 10)
  -a ADAPTER_MODE, --adapter_mode ADAPTER_MODE
                        Types of AF: (0.AF-Twin, 1.AF-EEG)
  -f FINETUNING, --finetuning FINETUNING
                        Finetuning on the target data or not
```
## Start Only Evaluation
If you don't need to finetune the network on the target data, please remove '-f' argument from the command line.
* '-f' argument is whether to finetune on the target data or not.
```bash
$ python -m Main -s 1 -a 0

subject: 1 | Top-1 Accuracy: 67.0 |Top-5 Accuracy:89.5
```
