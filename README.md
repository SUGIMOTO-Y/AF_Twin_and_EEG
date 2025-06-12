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

***
## Requirements
* Ubuntu==20.04.6 LTS
* Python==3.8.10
* CUDA==12.4
### Install
* pytorch==2.2.2
* numpy==1.24.4
* omegaconf==2.3.0

***
## Start
```bash
$ python -m Main -s 1 -a 0
```

```bash
$ python -m Main -h
usage: Main.py [-h] [-s SUBJECT] [-a ADAPTER_MODE]

optional arguments:
  -s The target subject (from 1 to 10)
  -a Types of AF: (0.AF-Twin, 1.AF-EEG)
```

