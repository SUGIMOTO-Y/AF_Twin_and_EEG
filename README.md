## AF_Twin_and_EEG
This is the Pytorch implementation of this paper: **Cross-subject EEG-based Visual Object Recognition: A Contrastive and Transfer Learning Approach**
	
Yuma Sugimoto<sup>1</sup>, Genci Capi<sup>2</sup>

<sup>1</sup>the Graudate School of Science and Engineering, Hosei University, Koganei, Tokyo, Japan.

<sup>2</sup>the Department of Mechanical Engineering, Hosei University, Koganei, Tokyo, Japan.
***
# Requirements
* Ubuntu==20.04.6 LTS
* Python==3.8.10
* CUDA==12.4
## Install
* pytorch==2.2.2
* numpy==1.24.4
* omegaconf==2.3.0

***
Start
'''
python -m Main -s 1 -a 0
'''

'''
usage: Main.py [-h] [-s SUBJECT] [-a ADAPTER_MODE]

optional arguments:
  -s The target subject (from 1 to 10)
  -a Types of AF: (0.AF-Twin, 1.AF-EEG)
'''

