import os
from omegaconf import OmegaConf
import torch
from utilities.config.Read_config import *

def set_args(args_):
    ModelConf = os.path.join('utilities', 'config', 'ModelConfig.yml')
    TrainConf = os.path.join('utilities', 'config', 'Finetune_Adapter_Config.yml')
    args = OmegaConf.merge(read_yaml(ModelConf), read_yaml(TrainConf))
    args.device =  'cuda:0' if torch.cuda.is_available() else 'cpu' 
    args.TargetSub = args_.subject
    args.Encoder_mode = 'both' if args_.adapter_mode == 0 else 'eeg'
    return args