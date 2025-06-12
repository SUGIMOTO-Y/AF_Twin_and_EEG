import torch
from utilities.DataLoader import *

def load_dataloader(args):
    load_dataset_ = get_dataset(args)
    test_eeg = load_dataset_.get_eeg
    test_img = load_dataset_.get_img        
    for i, (sig, _) in enumerate(test_eeg):
        if i == 0:
            print('EEG shape: {}'\
                .format(sig.cpu().numpy().shape))
            args.Model.SigSize = sig.cpu().numpy().shape
    args.Model.ImgSize = test_img.shape
    return  args, test_eeg, test_img